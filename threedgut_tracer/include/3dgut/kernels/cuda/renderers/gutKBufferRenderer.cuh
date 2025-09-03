// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <3dgut/kernels/cuda/common/rayPayloadBackward.cuh>
#include <3dgut/renderer/gutRendererParameters.h>

struct HitParticle {
    static constexpr float InvalidHitT = -1.0f;
    int idx                            = -1;
    float hitT                         = InvalidHitT;
    float alpha                        = 0.0f;
};

template <int K>
struct HitParticleKBuffer {
    __device__ HitParticleKBuffer() {
        m_numHits = 0;
#pragma unroll
        for (int i = 0; i < K; ++i) {
            m_kbuffer[i] = HitParticle();
        }
    }

    // insert a new hit into the kbuffer.
    // if the buffer is full overwrite the closest entry
    inline __device__ void insert(HitParticle& hitParticle) {
        const bool isFull = full();
        if (isFull) {
            m_kbuffer[0].hitT = HitParticle::InvalidHitT;
        } else {
            m_numHits++;
        }
#pragma unroll
        for (int i = K - 1; i >= 0; --i) {
            if (hitParticle.hitT > m_kbuffer[i].hitT) {
                const HitParticle tmp = m_kbuffer[i];
                m_kbuffer[i]          = hitParticle;
                hitParticle           = tmp;
            }
        }
    }

    inline __device__ const HitParticle& operator[](int i) const {
        return m_kbuffer[i];
    }

    inline __device__ uint32_t numHits() const {
        return m_numHits;
    }

    inline __device__ bool full() const {
        return m_numHits == K;
    }

    inline __device__ const HitParticle& closestHit(const HitParticle&) const {
        return m_kbuffer[0];
    }

private:
    HitParticle m_kbuffer[K];
    uint32_t m_numHits;
};

template <>
struct HitParticleKBuffer<0> {
    constexpr inline __device__ void insert(HitParticle& hitParticle) const {}
    constexpr inline __device__ HitParticle operator[](int) const { return HitParticle(); }
    constexpr inline __device__ uint32_t numHits() const { return 0; }
    constexpr inline __device__ bool full() const { return true; }
    constexpr inline __device__ const HitParticle& closestHit(const HitParticle& hitParticle) const { return hitParticle; }
};

template <typename Particles, typename Params, bool Backward = false>
struct GUTKBufferRenderer : Params {

    using DensityParameters    = typename Particles::DensityParameters;
    using DensityRawParameters = typename Particles::DensityRawParameters;
    using TFeaturesVec         = typename Particles::TFeaturesVec;

    using TRayPayload         = RayPayload<Particles::FeaturesDim, Particles::ExtendedFeaturesDim>;
    using TRayPayloadBackward = RayPayloadBackward<Particles::FeaturesDim, Particles::ExtendedFeaturesDim>;

    struct PrefetchedParticleData {
        uint32_t idx;
        DensityParameters densityParameters;
    };

    struct PrefetchedFullParticleData {
        uint32_t idx;
        TFeaturesVec features;
        DensityParameters densityParameters;
        tcnn::vec4 quaternion;
    };

    template <typename TRayPayload>
    static inline __device__ void processHitParticle(
        TRayPayload& ray,
        const HitParticle& hitParticle,
        const Particles& particles,
        const TFeaturesVec* __restrict__ particleFeatures,
        TFeaturesVec* __restrict__ particleFeaturesGradient) {

        if constexpr (Backward) {
            float hitAlphaGrad = 0.f;
            if constexpr (Params::PerRayParticleFeatures) {
                particles.featuresIntegrateBwdToBuffer<false>(ray.direction,
                                                              hitParticle.alpha,
                                                              hitAlphaGrad,
                                                              hitParticle.idx,
                                                              particles.featuresFromBuffer(hitParticle.idx, ray.direction),
                                                              threedgut::sliceVec<0, TRayPayload::BaseFeatDim>(ray.featuresBackward),
                                                              threedgut::sliceVec<0, TRayPayload::BaseFeatDim>(ray.featuresGradient));
            } else {
                TFeaturesVec particleFeaturesGradientVec = TFeaturesVec::zero();
                particles.featuresIntegrateBwd(hitParticle.alpha,
                                               hitAlphaGrad,
                                               particleFeatures[hitParticle.idx],
                                               particleFeaturesGradientVec,
                                               threedgut::sliceVec<0, TRayPayload::BaseFeatDim>(ray.featuresBackward),
                                               threedgut::sliceVec<0, TRayPayload::BaseFeatDim>(ray.featuresGradient));
#pragma unroll
                for (int i = 0; i < Particles::FeaturesDim; ++i) {
                    atomicAdd(&(particleFeaturesGradient[hitParticle.idx][i]), particleFeaturesGradientVec[i]);
                }
            }

            if constexpr (Particles::HasExtendedFeatures) {
                particles.extendedFeaturesIntegrateBwdToBuffer<false>(hitParticle.alpha,
                                                                      hitAlphaGrad,
                                                                      hitParticle.idx,
                                                                      threedgut::sliceVec<TRayPayload::BaseFeatDim, TRayPayload::ExtFeatDim>(ray.featuresBackward),
                                                                      threedgut::sliceVec<TRayPayload::BaseFeatDim, TRayPayload::ExtFeatDim>(ray.featuresGradient));
            }

            // Include normal gradients in backward pass
            tcnn::vec3 normal, normalGrad;
            particles.densityProcessHitBwdToBuffer<false>(ray.origin,
                                                          ray.direction,
                                                          hitParticle.idx,
                                                          hitParticle.alpha,
                                                          hitAlphaGrad,
                                                          ray.transmittanceBackward,
                                                          ray.transmittanceGradient,
                                                          hitParticle.hitT,
                                                          ray.hitTBackward,
                                                          ray.hitTGradient,
                                                          &normal,
                                                          &ray.integratedNormal,
                                                          &normalGrad);

            ray.transmittance *= (1.0 - hitParticle.alpha);

        } else {
            // Enable normal computation by passing normal pointer
            tcnn::vec3 normal;
            const float hitWeight =
                particles.densityIntegrateHit(hitParticle.alpha,
                                              ray.transmittance,
                                              hitParticle.hitT,
                                              ray.hitT,
                                              &normal,
                                              &ray.integratedNormal);

            if constexpr (Particles::HasExtendedFeatures) {
                particles.extendedFeaturesIntegrateFwdFromBuffer(hitWeight,
                                                                 hitParticle.idx,
                                                                 threedgut::sliceVec<TRayPayload::BaseFeatDim, TRayPayload::ExtFeatDim>(ray.features));
            }

            particles.featureIntegrateFwd(hitWeight,
                                          Params::PerRayParticleFeatures ? particles.featuresFromBuffer(hitParticle.idx, ray.direction) : particleFeatures[hitParticle.idx],
                                          threedgut::sliceVec<0, TRayPayload::BaseFeatDim>(ray.features));

            if (hitWeight > 0.0f)
                ray.countHit();
        }

        if (ray.transmittance < Particles::MinTransmittanceThreshold) {
            ray.kill();
        }
    }

    template <typename TRay>
    static inline __device__ void eval(const threedgut::RenderParameters& params,
                                       TRay& ray,
                                       const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                                       const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                       const tcnn::vec2* __restrict__ /*particlesProjectedPositionPtr*/,
                                       const tcnn::vec4* __restrict__ /*particlesProjectedConicOpacityPtr*/,
                                       const float* __restrict__ /*particlesGlobalDepthPtr*/,
                                       const float* __restrict__ particlesPrecomputedFeaturesPtr,
                                       threedgut::MemoryHandles parameters,
                                       tcnn::vec2* __restrict__ /*particlesProjectedPositionGradPtr*/     = nullptr,
                                       tcnn::vec4* __restrict__ /*particlesProjectedConicOpacityGradPtr*/ = nullptr,
                                       float* __restrict__ /*particlesGlobalDepthGradPtr*/                = nullptr,
                                       float* __restrict__ particlesPrecomputedFeaturesGradPtr            = nullptr,
                                       threedgut::MemoryHandles parametersGradient                        = {}) {

        using namespace threedgut;

        const uint32_t tileIdx                       = blockIdx.y * gridDim.x + blockIdx.x;
        const uint32_t tileThreadIdx                 = threadIdx.y * blockDim.x + threadIdx.x;
        const tcnn::uvec2 tileParticleRangeIndices   = sortedTileRangeIndicesPtr[tileIdx];
        uint32_t tileNumParticlesToProcess           = tileParticleRangeIndices.y - tileParticleRangeIndices.x;
        const uint32_t tileNumBlocksToProcess        = tcnn::div_round_up(tileNumParticlesToProcess, GUTParameters::Tiling::BlockSize);
        const TFeaturesVec* particleFeaturesBuffer   = Params::PerRayParticleFeatures ? nullptr : reinterpret_cast<const TFeaturesVec*>(particlesPrecomputedFeaturesPtr);
        TFeaturesVec* particleFeaturesGradientBuffer = (Params::PerRayParticleFeatures || !Backward) ? nullptr : reinterpret_cast<TFeaturesVec*>(particlesPrecomputedFeaturesGradPtr);

        Particles particles;
        particles.initializeDensity(parameters);
        if constexpr (Backward) {
            particles.initializeDensityGradient(parametersGradient);
        }
        particles.initializeFeatures(parameters);
        if constexpr (Backward && Params::PerRayParticleFeatures) {
            particles.initializeFeaturesGradient(parametersGradient);
        }
        if constexpr (Particles::HasExtendedFeatures) {
            particles.initializeExtendedFeatures(parameters);
            if constexpr (Backward) {
                particles.initializeExtendedFeaturesGradient(parametersGradient);
            }
        }

        if constexpr (false && Backward && (Params::KHitBufferSize == 0)) {
            evalBackwardNoKBuffer(ray, particles, tileParticleRangeIndices, tileNumBlocksToProcess, tileNumParticlesToProcess, tileThreadIdx,
                                  sortedTileParticleIdxPtr, particleFeaturesBuffer, particleFeaturesGradientBuffer);
        } else {
            evalKBuffer(ray, particles, tileParticleRangeIndices, tileNumBlocksToProcess, tileNumParticlesToProcess, tileThreadIdx,
                        sortedTileParticleIdxPtr, particleFeaturesBuffer, particleFeaturesGradientBuffer);
        }
    }

    template <typename TRay>
    static inline __device__ void evalKBuffer(TRay& ray,
                                              Particles& particles,
                                              const tcnn::uvec2& tileParticleRangeIndices,
                                              uint32_t tileNumBlocksToProcess,
                                              uint32_t tileNumParticlesToProcess,
                                              const uint32_t tileThreadIdx,
                                              const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                              const TFeaturesVec* __restrict__ particleFeaturesBuffer,
                                              TFeaturesVec* __restrict__ particleFeaturesGradientBuffer) {
        using namespace threedgut;
        __shared__ PrefetchedParticleData prefetchedParticlesData[GUTParameters::Tiling::BlockSize];

        HitParticleKBuffer<Params::KHitBufferSize> hitParticleKBuffer;

        for (uint32_t i = 0; i < tileNumBlocksToProcess; i++, tileNumParticlesToProcess -= GUTParameters::Tiling::BlockSize) {

            if (__syncthreads_and(!ray.isAlive())) {
                break;
            }

            // Collectively fetch particle data
            const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + i * GUTParameters::Tiling::BlockSize + tileThreadIdx;
            if (toProcessSortedIndex < tileParticleRangeIndices.y) {
                const uint32_t particleIdx = sortedTileParticleIdxPtr[toProcessSortedIndex];
                if (particleIdx != GUTParameters::InvalidParticleIdx) {
                    prefetchedParticlesData[tileThreadIdx] = {particleIdx, particles.fetchDensityParameters(particleIdx)};
                } else {
                    prefetchedParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
                }
            } else {
                prefetchedParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
            }
            __syncthreads();

            // Process fetched particles
            for (int j = 0; ray.isAlive() && j < min(GUTParameters::Tiling::BlockSize, tileNumParticlesToProcess); j++) {

                const PrefetchedParticleData particleData = prefetchedParticlesData[j];
                if (particleData.idx == GUTParameters::InvalidParticleIdx) {
                    i = tileNumBlocksToProcess;
                    break;
                }

                HitParticle hitParticle;
                hitParticle.idx = particleData.idx;
                if (particles.densityHit(ray.origin,
                                         ray.direction,
                                         particleData.densityParameters,
                                         hitParticle.alpha,
                                         hitParticle.hitT) &&
                    (hitParticle.hitT > ray.tMinMax.x) &&
                    (hitParticle.hitT < ray.tMinMax.y)) {

                    if (hitParticleKBuffer.full()) {
                        processHitParticle(ray,
                                           hitParticleKBuffer.closestHit(hitParticle),
                                           particles,
                                           particleFeaturesBuffer,
                                           particleFeaturesGradientBuffer);
                    }
                    hitParticleKBuffer.insert(hitParticle);
                }
            }
        }

        if constexpr (Params::KHitBufferSize > 0) {
            for (int i = 0; ray.isAlive() && (i < hitParticleKBuffer.numHits()); ++i) {
                processHitParticle(ray,
                                   hitParticleKBuffer[Params::KHitBufferSize - hitParticleKBuffer.numHits() + i],
                                   particles,
                                   particleFeaturesBuffer,
                                   particleFeaturesGradientBuffer);
            }
        }
    }

    template <typename TRay>
    static inline __device__ void evalBackwardNoKBuffer(TRay& ray,
                                                        Particles& particles,
                                                        const tcnn::uvec2& tileParticleRangeIndices,
                                                        uint32_t tileNumBlocksToProcess,
                                                        uint32_t tileNumParticlesToProcess,
                                                        const uint32_t tileThreadIdx,
                                                        const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                                        const TFeaturesVec* __restrict__ particleFeaturesBuffer,
                                                        TFeaturesVec* __restrict__ particleFeaturesGradientBuffer) {
        static_assert(Backward && (Params::KHitBufferSize == 0), "Optimized path for backward pass with no KBuffer");

        using namespace threedgut;
        __shared__ PrefetchedFullParticleData prefetchedFullParticlesData[GUTParameters::Tiling::BlockSize];

        for (uint32_t i = 0; i < tileNumBlocksToProcess; i++, tileNumParticlesToProcess -= GUTParameters::Tiling::BlockSize) {

            if (__syncthreads_and(!ray.isAlive())) {
                break;
            }

            // Collectively fetch particle data
            const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + i * GUTParameters::Tiling::BlockSize + tileThreadIdx;
            if (toProcessSortedIndex < tileParticleRangeIndices.y) {
                const uint32_t particleIdx = sortedTileParticleIdxPtr[toProcessSortedIndex];
                if (particleIdx != GUTParameters::InvalidParticleIdx) {
                    DensityRawParameters densityRawParameters                    = particles.fetchDensityRawParameters(particleIdx);
                    prefetchedFullParticlesData[tileThreadIdx].densityParameters = particles.densityParametersFromRaw(densityRawParameters);
                    prefetchedFullParticlesData[tileThreadIdx].quaternion =
                        {densityRawParameters.quaternion.x, densityRawParameters.quaternion.y, densityRawParameters.quaternion.z, densityRawParameters.quaternion.w};
                    if constexpr (Params::PerRayParticleFeatures) {
                        prefetchedFullParticlesData[tileThreadIdx].features = TFeaturesVec::zero();
                    } else {
                        prefetchedFullParticlesData[tileThreadIdx].features = particleFeaturesBuffer[particleIdx];
                    }
                    prefetchedFullParticlesData[tileThreadIdx].idx = particleIdx;
                } else {
                    prefetchedFullParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
                }
            } else {
                prefetchedFullParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
            }
            __syncthreads();

            // Process fetched particles
            for (int j = 0; j < min(GUTParameters::Tiling::BlockSize, tileNumParticlesToProcess); j++) {

                if (__all_sync(GUTParameters::Tiling::WarpMask, !ray.isAlive())) {
                    break;
                }

                const PrefetchedFullParticleData particleData = prefetchedFullParticlesData[j];
                if (particleData.idx == GUTParameters::InvalidParticleIdx) {
                    ray.kill();
                    break;
                }

                DensityRawParameters densityRawParametersGrad;
                densityRawParametersGrad.density    = 0.0f;
                densityRawParametersGrad.position   = tcnn::vec3(0.0f);
                densityRawParametersGrad.quaternion = tcnn::vec4(0.0f);
                densityRawParametersGrad.scale      = tcnn::vec3(0.0f);

                auto featuresGrad = tcnn::vec<Particles::FeaturesDim + Particles::ExtendedFeaturesDim>::zero();
                static_assert(TRayPayload::BaseFeatDim == Particles::FeaturesDim, "FeaturesDim mismatch");
                static_assert(TRayPayload::ExtFeatDim == Particles::ExtendedFeaturesDim, "ExtendedFeaturesDim mismatch");

                HitParticle hitParticle;
                hitParticle.idx = particleData.idx;

                if (ray.isAlive() &&
                    particles.densityHit(ray.origin,
                                         ray.direction,
                                         // FIXME : correct sensor specific support (eg LIDAR)
                                         Params::MinProjectedRayRadius * ray.spread,
                                         particleData.densityParameters,
                                         hitParticle.alpha,
                                         hitParticle.hitT) &&
                    (hitParticle.hitT > ray.tMinMax.x) &&
                    (hitParticle.hitT < ray.tMinMax.y)) {

                    float hitAlphaGrad = 0.f;
                    if constexpr (Params::PerRayParticleFeatures) {
                        // FIXME : support atomic warp sum for per ray particle features
                        particles.featuresIntegrateBwdToBuffer<false>(ray.direction,
                                                                      ray.directionGradient,
                                                                      hitParticle.alpha,
                                                                      hitAlphaGrad,
                                                                      hitParticle.idx,
                                                                      particles.featuresFromBuffer(hitParticle.idx, ray.direction),
                                                                      sliceVec<0, TRayPayload::BaseFeatDim>(ray.features),
                                                                      sliceVec<0, TRayPayload::BaseFeatDim>(ray.featuresGradient));
                    } else {
                        particles.featuresIntegrateBwd(hitParticle.alpha,
                                                       hitAlphaGrad,
                                                       particleFeaturesBuffer[hitParticle.idx],
                                                       sliceVec<0, Particles::FeaturesDim>(featuresGrad),
                                                       sliceVec<0, TRayPayload::BaseFeatDim>(ray.features),
                                                       sliceVec<0, TRayPayload::BaseFeatDim>(ray.featuresGradient));
                    }
                    if constexpr (Params::HasExtendedFeatures) {
                        particles.extendedFeaturesIntegrateBwdToVec(hitParticle.alpha,
                                                                    hitAlphaGrad,
                                                                    hitParticle.idx,
                                                                    particles.extendedFeaturesFromBuffer(hitParticle.idx),
                                                                    sliceVec<TRayPayload::BaseFeatDim, TRayPayload::ExtFeatDim>(ray.features),
                                                                    sliceVec<TRayPayload::BaseFeatDim, TRayPayload::ExtFeatDim>(ray.featuresGradient),
                                                                    sliceVec<TRayPayload::BaseFeatDim, TRayPayload::ExtFeatDim>(featuresGrad));
                    }

                    particles.densityProcessHitBwdToRawParameters(ray.origin,
                                                                  ray.originGradient,
                                                                  ray.direction,
                                                                  ray.directionGradient,
                                                                  // FIXME : correct sensor specific support (eg LIDAR)
                                                                  Params::MinProjectedRayRadius * ray.spread,
                                                                  particleData.idx,
                                                                  hitParticle.alpha,
                                                                  hitAlphaGrad,
                                                                  ray.transmittanceBackward,
                                                                  ray.transmittanceGradient,
                                                                  hitParticle.hitT,
                                                                  ray.hitT,
                                                                  ray.hitTGradient,
                                                                  particleData.densityParameters,
                                                                  particleData.quaternion,
                                                                  densityRawParametersGrad);

                    ray.transmittance *= (1.0f - hitParticle.alpha);
                    if (ray.transmittance < Particles::MinTransmittanceThreshold) {
                        ray.kill();
                    }
                }

                if constexpr (!Params::PerRayParticleFeatures) {
                    particles.processHitBwdUpdateFeaturesGradient<Particles::FeaturesDim>(
                        particleData.idx,
                        sliceVec<0, Particles::FeaturesDim>(featuresGrad),
                        particleFeaturesGradientBuffer,
                        tileThreadIdx);
                }
                if constexpr (Params::HasExtendedFeatures) {
                    particles.processHitBwdUpdateExtendedFeaturesGradientToBuffer(
                        particleData.idx,
                        sliceVec<Particles::FeaturesDim, Particles::ExtendedFeaturesDim>(featuresGrad),
                        tileThreadIdx);
                }
                particles.processHitBwdUpdateDensityGradient(particleData.idx, densityRawParametersGrad, tileThreadIdx);
            }
        }
    }
};
