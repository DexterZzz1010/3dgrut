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

namespace extendedFeatureParticle
{

struct ParametersBuffer
{
    const float* _dataPtr;
    float*_gradPtr;
    bool exclusiveGradient; //< true if the gradient maybe updated without atomics
};

struct Parameters : IDifferentiable
{
    float features[ExtendedFeaturesDim];
};

[BackwardDifferentiable][ForceInline]
Parameters fetchParametersFromBuffer(no_diff uint32_t particleIdx,
                                     no_diff ParametersBuffer parametersBuffer) 
{
    Parameters parameters;
    const uint32_t particleOffset = particleIdx * ExtendedFeaturesDim;
    [ForceUnroll] for (int i = 0; i < ExtendedFeaturesDim; ++i) {
        parameters.features[i] = parametersBuffer._dataPtr[particleOffset + i];
    }
    return parameters;
}

[BackwardDerivativeOf(fetchParametersFromBuffer)][ForceInline] 
void fetchParametersFromBufferBwd(no_diff uint32_t particleIdx,
                                  no_diff ParametersBuffer parametersBuffer,
                                  Parameters parametersGrad) 
{
    const uint32_t particleOffset = particleIdx * ExtendedFeaturesDim;
    if (parametersBuffer.exclusiveGradient) {
        [ForceUnroll] for (int i = 0; i < ExtendedFeaturesDim; ++i) {
            parametersBuffer._gradPtr[particleOffset + i] += parametersGrad.features[i];
        }
    } else {
        [ForceUnroll] for (int i = 0; i < ExtendedFeaturesDim; ++i) {
            InterlockedAdd(parametersBuffer._gradPtr[particleOffset + i], parametersGrad.features[i]);
        }
    }
}

[BackwardDifferentiable][ForceInline]
void integrateFeaturesFromParameters<let backToFront : bool>(float weight,
                                                             in Parameters parameters,
                                                             inout float integratedFeatures[ExtendedFeaturesDim])
{
    if (weight > 0.0f)
    {
        if (backToFront)
        {
            [ForceUnroll] for (int i = 0; i < ExtendedFeaturesDim; ++i) {
                integratedFeatures[i] = lerp(integratedFeatures[i], parameters.features[i], weight);
            }
        }
        else
        {
            [ForceUnroll] for (int i = 0; i < ExtendedFeaturesDim; ++i) {
                integratedFeatures[i] += parameters.features[i] * weight;
            }
        }
    }
}

[BackwardDifferentiable][ForceInline]
void integrateFeaturesFromBuffer<let backToFront : bool>(float weight,
                                                         no_diff uint32_t particleIdx,
                                                         no_diff ParametersBuffer parametersBuffer,
                                                         inout float integratedFeatures[ExtendedFeaturesDim])
{
    integrateFeaturesFromParameters<backToFront>(weight,
                                                 fetchParametersFromBuffer(particleIdx, parametersBuffer),
                                                 integratedFeatures);
}

} // namespace shRadiativeParticle

// ------------------------------------------------------------------------------------------------------------------
// Entry points

[CudaDeviceExport] inline void particleExtendedFeaturesIntegrateFwdFromBuffer(in float weight,
                                                           in uint32_t particleIdx,
                                                           extendedFeatureParticle.ParametersBuffer parametersBuffer,
                                                           inout float integratedFeatures[extendedFeatureParticle.ExtendedFeaturesDim])
{
    extendedFeatureParticle.integrateFeaturesFromBuffer<false>(weight, particleIdx, parametersBuffer, integratedFeatures);
}

[CudaDeviceExport] void particleExtendedFeaturesIntegrateBwdToBuffer(
    in float alpha,
    inout float alphaGrad,
    in uint32_t particleIdx,
    extendedFeatureParticle.ParametersBuffer parametersBuffer,
    inout float integratedFeatures[extendedFeatureParticle.ExtendedFeaturesDim],
    inout float integratedFeaturesGrad[extendedFeatureParticle.ExtendedFeaturesDim]) 
{
    if (alpha > 0.0f)
    {
        DifferentialPair<float> alphaDiff = DifferentialPair<float>(alpha, alphaGrad);

        const float weight = 1.0f / (1.0f - alpha);
        const extendedFeatureParticle.Parameters parameters = extendedFeatureParticle.fetchParametersFromBuffer(particleIdx, parametersBuffer);
        [ForceUnroll] for (int i = 0; i < extendedFeatureParticle.ExtendedFeaturesDim; ++i) {
            integratedFeatures[i] = (integratedFeatures[i] - parameters.features[i] * alpha) * weight;
        }
        DifferentialPair<float[extendedFeatureParticle.ExtendedFeaturesDim]> integratedFeaturesDiff = 
            DifferentialPair<float[extendedFeatureParticle.ExtendedFeaturesDim]>(integratedFeatures, integratedFeaturesGrad);

        bwd_diff(extendedFeatureParticle.integrateFeaturesFromBuffer<true>)(
            alphaDiff,
            particleIdx,
            parametersBuffer,
            integratedFeaturesDiff);

        integratedFeaturesGrad = integratedFeaturesDiff.getDifferential();
        alphaGrad += alphaDiff.getDifferential();
    }
}
