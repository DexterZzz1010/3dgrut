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

#include <3dgut/kernels/cuda/common/rayPayload.cuh>

template <int BaseFeatN = 3, int ExtFeatN = 0>
struct RayPayloadBackward : public RayPayload<BaseFeatN, ExtFeatN> {
    float transmittanceBackward;
    float transmittanceGradient;
    float hitTBackward;
    float hitTGradient;
    tcnn::vec<BaseFeatN + ExtFeatN> featuresGradient;
    tcnn::vec<BaseFeatN + ExtFeatN> featuresBackward;
};

template <typename RayPayloadT>
__device__ __inline__ RayPayloadT initializeBackwardRay(const threedgut::RenderParameters& params,
                                                        const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                                                        const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
                                                        const float* __restrict__ worldHitDistancePtr,
                                                        const float* __restrict__ worldHitDistanceGradientPtr,
                                                        const tcnn::vec<RayPayloadT::BaseFeatDim + 1>* __restrict__ featuresDensityPtr,
                                                        const tcnn::vec<RayPayloadT::BaseFeatDim + 1>* __restrict__ featuresDensityGradientPtr,
                                                        const float* __restrict__ extendedFeaturesPtr,
                                                        const float* __restrict__ extendedFeaturesGradientPtr,
                                                        const tcnn::mat4x3& sensorToWorldTransform) {

    // NB : no backpropagation through the forward ray initialization / finalization
    RayPayloadT ray = initializeRay<RayPayloadT>(params,
                                                 sensorRayOriginPtr,
                                                 sensorRayDirectionPtr,
                                                 sensorToWorldTransform);

    if (ray.isAlive()) {
        const tcnn::vec<RayPayloadT::FeatDim + 1> featuresDensity              = featuresDensityPtr[ray.idx];
        const tcnn::vec<RayPayloadT::FeatDim + 1> featuresDensityGradient      = featuresDensityGradientPtr[ray.idx];
        ray.transmittanceBackward                                              = 1.f - featuresDensity[RayPayloadT::FeatDim];
        ray.transmittanceGradient                                              = -1.f * featuresDensityGradient[RayPayloadT::FeatDim];
        ray.hitTBackward                                                       = worldHitDistancePtr[ray.idx];
        ray.hitTGradient                                                       = worldHitDistanceGradientPtr[ray.idx];
        threedgut::sliceVec<0, RayPayloadT::BaseFeatDim>(ray.featuresBackward) = threedgut::sliceVec<0, RayPayloadT::BaseFeatDim>(featuresDensity);
        threedgut::sliceVec<0, RayPayloadT::BaseFeatDim>(ray.featuresGradient) = threedgut::sliceVec<0, RayPayloadT::BaseFeatDim>(featuresDensityGradient);
        if constexpr (RayPayloadT::ExtFeatDim > 0) {
            threedgut::sliceVec<RayPayloadT::BaseFeatDim, RayPayloadT::ExtFeatDim>(ray.featuresBackward) =
                reinterpret_cast<const tcnn::vec<RayPayloadT::ExtFeatDim>*>(extendedFeaturesPtr)[ray.idx];
            threedgut::sliceVec<RayPayloadT::BaseFeatDim, RayPayloadT::ExtFeatDim>(ray.featuresGradient) =
                reinterpret_cast<const tcnn::vec<RayPayloadT::ExtFeatDim>*>(extendedFeaturesGradientPtr)[ray.idx];
        }
    }

    return ray;
}
