Texture2D<float4> BaseTex : register(t0);
Texture2D<float4> Latent0Tex : register(t1);
Texture2D<float4> Latent1Tex : register(t2);
StructuredBuffer<float4> PhaseParams : register(t3);
SamplerState LinearSampler : register(s0);
RWTexture2D<float4> OutputTex : register(u0);
cbuffer Decoder : register(b0) {
    float4 W0a; float4 W0b;
    float4 W1a; float4 W1b;
    float4 W2a; float4 W2b;
    float4 DecoderBias;
    uint DecoderMode;
    uint LatentChannels;
    uint PhaseWeightCount;
    uint Reserved;
};
float4 LatentVectorAt(int2 position, uint textureIndex) {
    uint latentW, latentH;
    Latent0Tex.GetDimensions(latentW, latentH);
    position = clamp(position, int2(0, 0), int2(latentW - 1, latentH - 1));
    return textureIndex == 0 ? Latent0Tex.Load(int3(position, 0)) : Latent1Tex.Load(int3(position, 0));
}
float3 PhaseResidual(uint2 outputPosition) {
    uint phaseX = outputPosition.x % 3;
    uint phaseY = outputPosition.y % 3;
    int2 center = int2((outputPosition.x / 3) * 2 + (phaseX == 2 ? 1 : 0),
        (outputPosition.y / 3) * 2 + (phaseY == 2 ? 1 : 0));
    uint phase = phaseY * 3 + phaseX;
    float3 result = PhaseParams[PhaseWeightCount + phase].xyz;
    uint textureCount = (LatentChannels + 3) / 4;
    [loop] for (uint textureIndex = 0; textureIndex < textureCount; ++textureIndex) {
        [unroll] for (uint ky = 0; ky < 3; ++ky) {
            [unroll] for (uint kx = 0; kx < 3; ++kx) {
                float4 latent = LatentVectorAt(center + int2((int)kx - 1, (int)ky - 1), textureIndex);
                uint tapBase = (((phase * 3) * 3 + ky) * 3 + kx) * textureCount + textureIndex;
                uint rgbStride = 3 * 3 * textureCount;
                result.x += dot(latent, PhaseParams[tapBase]);
                result.y += dot(latent, PhaseParams[tapBase + rgbStride]);
                result.z += dot(latent, PhaseParams[tapBase + rgbStride * 2]);
            }
        }
    }
    return result;
}
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint outW, outH;
    OutputTex.GetDimensions(outW, outH);
    if (id.x >= outW || id.y >= outH) return;
    float2 uv = (float2(id.xy) + 0.5) / float2(outW, outH);
    float3 base = BaseTex.SampleLevel(LinearSampler, uv, 0).rgb;
    if (DecoderMode == 1) {
        float3 residual = PhaseResidual(id.xy);
        OutputTex[id.xy] = float4(saturate(base + residual), 1.0);
        return;
    }
    float4 latent0 = Latent0Tex.SampleLevel(LinearSampler, uv, 0);
    float4 latent1 = Latent1Tex.SampleLevel(LinearSampler, uv, 0);
    float3 residual = float3(dot(latent0, W0a) + dot(latent1, W0b) + DecoderBias.x,
        dot(latent0, W1a) + dot(latent1, W1b) + DecoderBias.y,
        dot(latent0, W2a) + dot(latent1, W2b) + DecoderBias.z);
    OutputTex[id.xy] = float4(saturate(base + residual), 1.0);
}
