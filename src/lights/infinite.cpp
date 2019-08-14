
/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

// lights/infinite.cpp*
#include "lights/infinite.h"
#include "imageio.h"
#include "paramset.h"
#include "reflection.h"
#include "sampling.h"
#include "stats.h"

namespace pbrt {

static Vector3f SurfaceLocalToWorld(const SurfaceInteraction &si, const Vector3f &v) {
    Normal3f ns = si.shading.n;
    Vector3f ss = Normalize(si.shading.dpdu);
    Vector3f ts = Cross(ns, ss);
    return Vector3f(ss.x * v.x + ts.x * v.y + ns.x * v.z,
                    ss.y * v.x + ts.y * v.y + ns.y * v.z,
                    ss.z * v.x + ts.z * v.y + ns.z * v.z);
}

// InfiniteAreaLight Method Definitions
InfiniteAreaLight::InfiniteAreaLight(const Transform &LightToWorld,
                                     const Spectrum &L, int nSamples,
                                     const std::string &texmap,
                                     SamplingMethod samplingMethod)
    : Light((int)LightFlags::Infinite, LightToWorld, MediumInterface(),
            nSamples), samplingMethod(samplingMethod) {
    // Read texel data from _texmap_ and initialize _Lmap_
    Point2i resolution;
    std::unique_ptr<RGBSpectrum[]> texels(nullptr);
    if (texmap != "") {
        texels = ReadImage(texmap, &resolution);
        if (texels)
            for (int i = 0; i < resolution.x * resolution.y; ++i)
                texels[i] *= L.ToRGBSpectrum();
    }
    if (!texels) {
        resolution.x = resolution.y = 1;
        texels = std::unique_ptr<RGBSpectrum[]>(new RGBSpectrum[1]);
        texels[0] = L.ToRGBSpectrum();
    }
    Lmap.reset(new MIPMap<RGBSpectrum>(resolution, texels.get()));

    // Initialize sampling PDFs for infinite area light

    if (samplingMethod == SamplingMethod::Distribution || samplingMethod == SamplingMethod::HierarchicalWarping) {
        // Compute scalar-valued image _img_ from environment map
        int width = 2 * Lmap->Width(), height = 2 * Lmap->Height();
        std::unique_ptr<Float[]> img(new Float[width * height]);
        float fwidth = 0.5f / std::min(width, height);
        ParallelFor(
            [&](int64_t v) {
                Float vp = (v + .5f) / (Float)height;
                Float sinTheta = std::sin(Pi * (v + .5f) / height);
                for (int u = 0; u < width; ++u) {
                    Float up = (u + .5f) / (Float)width;
                    img[u + v * width] =
                        Lmap->Lookup(Point2f(up, vp), fwidth).y();
                    img[u + v * width] *= sinTheta;
                }
            },
            height, 32);

        if (samplingMethod == SamplingMethod::Distribution) {
            // Compute sampling distributions for rows and columns of image
            distribution.reset(new Distribution2D(img.get(), width, height));
        } else {
            importanceMap.reset(new MIPMap<Float>(
                2 * resolution, img.get(), false, 0.0f, ImageWrap::Black));
        }
    }
}

Spectrum InfiniteAreaLight::Power() const {
    return Pi * worldRadius * worldRadius *
           Spectrum(Lmap->Lookup(Point2f(.5f, .5f), .5f),
                    SpectrumType::Illuminant);
}

Spectrum InfiniteAreaLight::Le(const RayDifferential &ray) const {
    Vector3f w = Normalize(WorldToLight(ray.d));
    Point2f st(SphericalPhi(w) * Inv2Pi, SphericalTheta(w) * InvPi);
    return Spectrum(Lmap->Lookup(st), SpectrumType::Illuminant);
}

Spectrum InfiniteAreaLight::Sample_Li(const Interaction &ref, const Point2f &u,
                                      Vector3f *wi, Float *pdf,
                                      VisibilityTester *vis) const {
    ProfilePhase _(Prof::LightSample);
    // Find $(u,v)$ sample coordinates in infinite light texture
    Point2f uv;
    if (samplingMethod == SamplingMethod::Hemisphere) {
        if (!ref.IsSurfaceInteraction()) {
            Error(
                "InfiniteAreaLight::Sample_Li \"hemisphere\" sampling method cannot "
                "handle non-surface interactions.");
            return Spectrum(0.f);
        }

        *wi = CosineSampleHemisphere(u);
        *pdf = CosineHemispherePdf(AbsCosTheta(*wi));
        *wi = SurfaceLocalToWorld(static_cast<const SurfaceInteraction &>(ref), *wi);

        Vector3f wiLight = WorldToLight(*wi);
        uv[0] = SphericalPhi(wiLight) / (2 * Pi);
        uv[1] = SphericalTheta(wiLight) / Pi;
    } else {
        Float mapPdf;
        if (samplingMethod == SamplingMethod::Uniform) {
            uv = u;
            mapPdf = 1.0f;
        }
        else if (samplingMethod == SamplingMethod::Distribution) {
            uv = distribution->SampleContinuous(u, &mapPdf);
        } else {
            uv = SampleImportanceMap(u, &mapPdf);
        }
        if (mapPdf == 0) return Spectrum(0.f);

        // Convert infinite light sample point to direction
        Float theta = uv[1] * Pi, phi = uv[0] * 2 * Pi;
        Float cosTheta = std::cos(theta), sinTheta = std::sin(theta);
        Float sinPhi = std::sin(phi), cosPhi = std::cos(phi);

        *wi = LightToWorld(
            Vector3f(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta));

        // Compute PDF for sampled infinite light direction
        *pdf = mapPdf / (2 * Pi * Pi * sinTheta);
        if (sinTheta == 0) *pdf = 0;
    }

    // Return radiance value for infinite light direction
    *vis = VisibilityTester(ref, Interaction(ref.p + *wi * (2 * worldRadius),
                                             ref.time, mediumInterface));
    return Spectrum(Lmap->Lookup(uv), SpectrumType::Illuminant);
}

Float InfiniteAreaLight::Pdf_Li(const Interaction &ref,
                                const Vector3f &w) const {
    ProfilePhase _(Prof::LightPdf);
    if( samplingMethod == SamplingMethod::Uniform ) {
        Vector3f wi = WorldToLight(w);
        Float sinTheta = std::sin(SphericalTheta(wi));
        if (sinTheta == 0) return 0;
        Float mapPdf = 1.0f;
        return mapPdf / (2 * Pi * Pi * sinTheta);
    }
    if (samplingMethod == SamplingMethod::Hemisphere) {
        Float cosTheta = AbsDot(ref.n, w);
        return CosineHemispherePdf(cosTheta);
    } else {
        Vector3f wi = WorldToLight(w);
        Float theta = SphericalTheta(wi), phi = SphericalPhi(wi);
        Float sinTheta = std::sin(theta);
        if (sinTheta == 0) return 0;
        if (samplingMethod == SamplingMethod::Distribution) {
            return distribution->Pdf(Point2f(phi * Inv2Pi, theta * InvPi)) /
                   (2 * Pi * Pi * sinTheta);
        } else {
            int maxLevel = importanceMap->Levels() - 1;
            int s = std::floor((phi * Inv2Pi) * importanceMap->Width() - 0.5f);
            int t = std::floor((theta * InvPi) * importanceMap->Height() - 0.5f);
            Float mapPdf = importanceMap->Texel(0, s, t) /
                           importanceMap->Texel(maxLevel, 0, 0);
            return mapPdf / (2 * Pi * Pi * sinTheta);
        }
    }
}

Spectrum InfiniteAreaLight::Sample_Le(const Point2f &u1, const Point2f &u2,
                                      Float time, Ray *ray, Normal3f *nLight,
                                      Float *pdfPos, Float *pdfDir) const {
    if (samplingMethod != SamplingMethod::Distribution) {
        Error(
            "InfiniteAreaLight::Sample_Le supports only \"distribution\" "
            "sampling method.");
        return Spectrum(0.f);
    }

    ProfilePhase _(Prof::LightSample);
    // Compute direction for infinite light sample ray
    Point2f u = u1;

    // Find $(u,v)$ sample coordinates in infinite light texture
    Float mapPdf;
    Point2f uv = distribution->SampleContinuous(u, &mapPdf);
    if (mapPdf == 0) return Spectrum(0.f);
    Float theta = uv[1] * Pi, phi = uv[0] * 2.f * Pi;
    Float cosTheta = std::cos(theta), sinTheta = std::sin(theta);
    Float sinPhi = std::sin(phi), cosPhi = std::cos(phi);
    Vector3f d =
        -LightToWorld(Vector3f(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta));
    *nLight = (Normal3f)d;

    // Compute origin for infinite light sample ray
    Vector3f v1, v2;
    CoordinateSystem(-d, &v1, &v2);
    Point2f cd = ConcentricSampleDisk(u2);
    Point3f pDisk = worldCenter + worldRadius * (cd.x * v1 + cd.y * v2);
    *ray = Ray(pDisk + worldRadius * -d, d, Infinity, time);

    // Compute _InfiniteAreaLight_ ray PDFs
    *pdfDir = sinTheta == 0 ? 0 : mapPdf / (2 * Pi * Pi * sinTheta);
    *pdfPos = 1 / (Pi * worldRadius * worldRadius);
    return Spectrum(Lmap->Lookup(uv), SpectrumType::Illuminant);
}

void InfiniteAreaLight::Pdf_Le(const Ray &ray, const Normal3f &, Float *pdfPos,
                               Float *pdfDir) const {
    if (samplingMethod != SamplingMethod::Distribution) {
        Error(
            "InfiniteAreaLight::Pdf_Le supports only \"distribution\" sampling "
            "method.");
        return;
    }

    ProfilePhase _(Prof::LightPdf);
    Vector3f d = -WorldToLight(ray.d);
    Float theta = SphericalTheta(d), phi = SphericalPhi(d);
    Point2f uv(phi * Inv2Pi, theta * InvPi);
    Float mapPdf = distribution->Pdf(uv);
    *pdfDir = mapPdf / (2 * Pi * Pi * std::sin(theta));
    *pdfPos = 1 / (Pi * worldRadius * worldRadius);
}

Point2f InfiniteAreaLight::SampleImportanceMap(Point2f u, Float *pdf) const {
    int s = 0, t = 0;
    int maxLevel = importanceMap->Levels() - 1;
    for (int level = maxLevel - 1; level >= 0; --level) {
        s <<= 1; t <<= 1;

        Float importance[4];
        importance[0] = importanceMap->Texel(level, s, t);
        importance[1] = importanceMap->Texel(level, s, t + 1);
        importance[2] = importanceMap->Texel(level, s + 1, t);
        importance[3] = importanceMap->Texel(level, s + 1, t + 1);

        Float marginalLeft  = importance[0] + importance[1];
        Float marginalRight = importance[2] + importance[3];

        float pConditionalLeft = marginalLeft / (marginalLeft + marginalRight);
        if (u[0] < pConditionalLeft) {
            u[0] /= pConditionalLeft;
            Float pConditionalLower = importance[1] / marginalLeft;
            if (u[1] < pConditionalLower) {
                ++t;
                u[1] /= pConditionalLower;
            } else
                u[1] = (u[1] - pConditionalLower) / (1.0f - pConditionalLower);
        } else {
            ++s;
            u[0] = (u[0] - pConditionalLeft) / (1.0f - pConditionalLeft);
            Float pConditionalLower = importance[3] / marginalRight;
            if (u[1] < pConditionalLower) {
                ++t;
                u[1] /= pConditionalLower;
            } else
                u[1] = (u[1] - pConditionalLower) / (1.0f - pConditionalLower);
        }
    }

    *pdf = importanceMap->Texel(0, s, t) / importanceMap->Texel(maxLevel, 0, 0);
    return Point2f((s + u[0]) / (Float)importanceMap->Width(), (t + u[1]) / (Float)importanceMap->Height());
}

std::shared_ptr<InfiniteAreaLight> CreateInfiniteLight(
    const Transform &light2world, const ParamSet &paramSet) {
    Spectrum L = paramSet.FindOneSpectrum("L", Spectrum(1.0));
    Spectrum sc = paramSet.FindOneSpectrum("scale", Spectrum(1.0));
    std::string texmap = paramSet.FindOneFilename("mapname", "");
    int nSamples =
        paramSet.FindOneInt("samples", paramSet.FindOneInt("nsamples", 1));
    std::string samplingMethodName =
        paramSet.FindOneString("samplingmethod", "distribution");
    InfiniteAreaLight::SamplingMethod samplingMethod =
        InfiniteAreaLight::SamplingMethod::Distribution;
    if (samplingMethodName == "uniform")
        samplingMethod = InfiniteAreaLight::SamplingMethod::Uniform;
    else if (samplingMethodName == "hemisphere" )
        samplingMethod = InfiniteAreaLight::SamplingMethod::Hemisphere;
    else if (samplingMethodName == "distribution")
        samplingMethod = InfiniteAreaLight::SamplingMethod::Distribution;
    else if (samplingMethodName == "hwarp")
        samplingMethod = InfiniteAreaLight::SamplingMethod::HierarchicalWarping;
    else {
        Warning(
            "Infinite area light sampling method \"%s\" unknown. Using "
            "\"distribution\".",
            samplingMethodName.c_str());
        samplingMethod = InfiniteAreaLight::SamplingMethod::Distribution;
    }
    if (PbrtOptions.quickRender) nSamples = std::max(1, nSamples / 4);
    return std::make_shared<InfiniteAreaLight>(light2world, L * sc, nSamples,
                                               texmap, samplingMethod);
}

}  // namespace pbrt
