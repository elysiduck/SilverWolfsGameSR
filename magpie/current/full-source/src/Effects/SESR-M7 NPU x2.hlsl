// The actual SESR-M7 x2 operation is performed by SesrNpuProcessor on the
// AMD XDNA2 NPU. This inexpensive pass exposes the backend as a normal Magpie
// effect and keeps it composable with sharpening/post-processing effects.

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
