// Deployment-diagnostic SESR-M7 mode. Renderer deliberately bypasses the
// Exact-1080p GPU pre-resize and sends the captured texture to the NPU at its
// current resolution. The fixed x2 model output is fitted to the display later.

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
