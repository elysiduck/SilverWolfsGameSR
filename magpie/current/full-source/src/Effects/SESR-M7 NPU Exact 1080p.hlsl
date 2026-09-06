// Exact-output SESR-M7 mode. Renderer performs a GPU Bicubic pre-resize to
// half of the aspect-fitted display area before the NPU x2 pass.

//!MAGPIE EFFECT
//!VERSION 4

//!TEXTURE
Texture2D INPUT;

//!TEXTURE
//!WIDTH INPUT_WIDTH
//!HEIGHT INPUT_HEIGHT
Texture2D OUTPUT;

//!SAMPLER
//!FILTER POINT
SamplerState sam;

//!PASS 1
//!STYLE PS
//!IN INPUT
//!OUT OUTPUT
float4 Pass1(float2 pos) {
	return INPUT.SampleLevel(sam, pos, 0);
}
