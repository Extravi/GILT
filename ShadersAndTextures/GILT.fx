/*
 *  Global illumination shader for Nvidia Ansel, written by Extravi.
 *  https://extravi.dev/
*/

#include "ReShade.fxh"

#define FAR_PLANE RESHADE_DEPTH_LINEARIZATION_FAR_PLANE

uniform float total_strength <
    ui_type = "slider";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_label = "Intensity";
> = 1.0;

uniform float RAY_INC <
    ui_type = "slider";
    ui_min = 1;
    ui_max = 8;
    ui_label = "Ray Increment";
> = 4.0;

// aka ray steps
uniform float RAY_LEN <
    ui_type = "slider";
    ui_min = 1;
    ui_max = 32;
    ui_label = "Ray Length";
> = 16;

uniform float INDIRECT_STRENGTH <
    ui_type = "slider";
    ui_min = 100.0;
    ui_max = 1000.0;
    ui_label = "Indirect Light Strength";
> = 500.0;

uniform bool DEBUG <
    ui_type = "bool";
    ui_label = "Debug";
> = false;

// used for the light map
uniform float2 SAT_EXP <
    ui_type = "slider";
    ui_min = 0.0;
    ui_max = 20.0;
    ui_label = "Saturation\n& Exposure";
> = float2(20.0, 0.0);

//////////////////////////////////////
// Textures and samplers
//////////////////////////////////////

// Light Map texture
texture LightMapTex {
    Width = BUFFER_WIDTH / 1;
    Height = BUFFER_HEIGHT / 1;
    MipLevels = 3;
};
// Buffer texture
texture BufferTex {
    Width = BUFFER_WIDTH / 1;
    Height = BUFFER_HEIGHT / 1;
    Format = R16F;
    MipLevels = 3;
};
// Normal texture
texture NormalTex {
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    MipLevels = 1;
};
// Output texture
texture OutputTex {
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA16F;
    MipLevels = 2;
};

// Light Map sampler
sampler LightMapSampler {
    Texture = LightMapTex;
    MipFilter = LINEAR;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
    AddressU = WRAP;
    AddressV = WRAP;
    AddressW = WRAP;
};
// Buffer sampler
sampler BufferSampler {
    Texture = BufferTex;
};
// Normal sampler
sampler NormalSampler {
    Texture = NormalTex;
};
// Output sampler
sampler OutputSampler {
    Texture = OutputTex;
};

//////////////////////////////////////
// Functions
//////////////////////////////////////

// function for SAT_EXP uniform
float3 AdjustColor(float3 color, float2 SAT_EXP)
{
    float saturation = SAT_EXP.x;
    float exposure = SAT_EXP.y;

    // apply exposure
    color *= pow(2.0, exposure);

    // convert to grayscale
    float gray = dot(color, float3(0.299, 0.587, 0.114));

    // adjust saturation
    return lerp(float3(gray, gray, gray), color, saturation);
}

// function to get the buffer color
float3 GetBackBufferColor(float2 texcoord)
{
    return tex2D(ReShade::BackBuffer, texcoord).rgb;
}

float3 PS_LightMapAdjust(float4 position : SV_Position, float2 texcoord : TEXCOORD0) : SV_Target
{
    float3 bufferColor = GetBackBufferColor(texcoord);
    return AdjustColor(bufferColor, SAT_EXP);
}

// function to get the depths buffer
float GetDepth(float2 texcoord)
{
    return ReShade::GetLinearizedDepth(texcoord);
}

float3 PS_DisplayDepth(float4 position : SV_Position, float2 texcoord : TEXCOORD0) : SV_Target
{
    // get the depth value at the texture coordinate
    float depth = GetDepth(texcoord);
    
    // normalize depth
    return depth;
}

float3 PS_NormalBuffer(float4 position : SV_Position, float2 texcoord : TEXCOORD0) : SV_Target
{
    // get the depth value at the texture coordinate
    float depth = GetDepth(texcoord);
    // buffer dimensions vector dims
    float2 dims = float2(BUFFER_WIDTH, BUFFER_HEIGHT);

    // horizontal differences
    float2 texOffset = float2(1, 0) / dims;
    float depthsX = depth - ReShade::GetLinearizedDepth(texcoord - texOffset);
    depthsX += (depth - ReShade::GetLinearizedDepth(texcoord + texOffset)) - depthsX;

    // vertical  differences
    texOffset = float2(0, 1) / dims;
    float depthsY = depth - ReShade::GetLinearizedDepth(texcoord - texOffset);
    depthsY += (depth - ReShade::GetLinearizedDepth(texcoord + texOffset)) - depthsY;

    // normalized normal
    return 0.5 + 0.5 * normalize(float3(depthsX, depthsY, depth / FAR_PLANE));
}

float3 eyePos(float2 xy, float z)
{
	z = -z;
    float3 eyp = float3(xy, 1 ) * z ;
    return eyp;
}

float4 PS_ComputeGI(float4 position : SV_Position, float2 coord : TEXCOORD0) : SV_Target
{
    // constants
    const int samples = 16;
    float2 dims = 1 / float2(BUFFER_WIDTH, BUFFER_HEIGHT);
    float3 sample_sphere[samples] = 
    {
        float3( 0.5381, 0.1856,-0.4319), float3( 0.1379, 0.2486, 0.4430),
        float3( 0.3371, 0.5679,-0.0057), float3(-0.6999,-0.0451,-0.0019),
        float3( 0.0689,-0.1598,-0.8547), float3( 0.0560, 0.0069,-0.1843),
        float3(-0.0146, 0.1402, 0.0762), float3( 0.0100,-0.1924,-0.0344),
        float3(-0.3577,-0.5301,-0.4358), float3(-0.3169, 0.1063, 0.0158),
        float3( 0.0103,-0.5869, 0.0046), float3(-0.0897,-0.4940, 0.3287),
        float3( 0.7119,-0.0154,-0.0918), float3(-0.0533, 0.0596,-0.5411),
        float3( 0.0352,-0.0631, 0.5460), float3(-0.4776, 0.2847,-0.0271)
    };

    // generate noise
    float3 Noise = frac(sin(dot(coord.xy , float2(12.9898,78.233))) * 43758.5453);

    // get depth and normal
    float depth = GetDepth(coord);
    float3 normal = normalize(PS_NormalBuffer(position, coord));

    // eye position
    float3 eye_position = eyePos(coord, depth);
    float3 ray_dir = normalize(eye_position);
    
    // ray increment
    float RayInc = RAY_INC * dims.x;

    float3 accumulatedColor = float3(0, 0, 0);
    float totalWeight = 0.0;

    for (int i = 0; i < samples; i++)
    {
        float3 sampleRay = reflect(sample_sphere[i], Noise);
        float3 rayPos = eye_position;

        for(int j = 0; j < RAY_LEN; j++)
        {
            rayPos += sampleRay * RayInc;

            // ensure sample coordinates are within valid range
            float2 sampleCoord = saturate((rayPos.xy / rayPos.z));
            if (sampleCoord.x < 0.0 || sampleCoord.x > 1.0 || sampleCoord.y < 0.0 || sampleCoord.y > 1.0)
                continue;

            float sampleDepth = tex2D(BufferSampler, sampleCoord).x;
            float3 sampleColor = tex2D(LightMapSampler, sampleCoord).rgb;
            
            // apply depth fade
            float depthDifference = abs(depth - sampleDepth);
            float fade = smoothstep(0.0, 1.0, depthDifference);
            
            // check depth difference with a small epsilon to handle precision issues
            float difference = abs(depth - sampleDepth);
            accumulatedColor += sampleColor * step(difference, lerp(0.0001f, 0.1f, total_strength)) * INDIRECT_STRENGTH * fade;
            totalWeight++;
        }
    }

    // average out the accumulated color
    float3 DebugColor = accumulatedColor / totalWeight;
    
    // apply a tone-mapping function to prevent oversaturation and over-brightness
    DebugColor = DebugColor / (DebugColor + 1.0);

    // apply a gamma correction
    DebugColor = pow(DebugColor, 1.0 / 2.2);

    // fix depths
    DebugColor = lerp(DebugColor, float3(0.5, 0.5, 0.5), depth);

    float3 finalColor = tex2D(LightMapSampler, coord).rgb;
    
    return float4(DebugColor, 1.0);
}

// blend backBufferColor and OutputSampler
float3 SoftLightBlend(float3 base, float3 blend)
{
    float3 result;
    for (int i = 0; i < 3; i++)
    {
        if (blend[i] < 0.5)
        {
            result[i] = base[i] - (1.0 - 2.0 * blend[i]) * base[i] * (1.0 - base[i]);
        }
        else
        {
            result[i] = base[i] + (2.0 * blend[i] - 1.0) * (sqrt(base[i]) - base[i]);
        }
    }
    return result;
}

// used to denoise
float3 GaussianBlur(sampler2D texSampler, float2 uv, float2 texSize)
{
    float3 blur = float3(0.0, 0.0, 0.0);

    const int radius = 5;
    const float sigma = 3.0;

    float2 texelSize = 1.0 / texSize;

    float kernel[11];
    float sum = 0.0;
    
    for (int i = -radius; i <= radius; ++i)
    {
        kernel[i + radius] = exp(-0.5 * (i * i) / (sigma * sigma));
        sum += kernel[i + radius];
    }

    for (int i = 0; i < 11; ++i)
    {
        kernel[i] /= sum;
    }

    for (int x = -radius; x <= radius; ++x)
    {
        for (int y = -radius; y <= radius; ++y)
        {
            float2 offset = float2(x, y) * texelSize;
            float weight = kernel[x + radius] * kernel[y + radius];
            blur += tex2D(texSampler, uv + offset).rgb * weight;
        }
    }

    return blur;
}

float4 Output(float4 position : SV_Position, float2 coord : TEXCOORD0) : SV_Target
{
    // sample the colors
    float3 backBufferColor = tex2D(ReShade::BackBuffer, coord).rgb;
    float3 finalColor = tex2D(OutputSampler, coord).rgb;

    // apply Gaussian blur
    float2 texSize = float2(BUFFER_WIDTH, BUFFER_HEIGHT);
    float3 blurredColor = GaussianBlur(OutputSampler, coord, texSize);

    // apply soft light blending
    float3 blendedColor = SoftLightBlend(backBufferColor, blurredColor);

    if (DEBUG == true)
    {
        return float4(blurredColor, 1.0);
    }
    else
    {
        return float4(blendedColor, 1.0);
    }
}

// Global Illumination Lighting Technique
technique GILT
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_LightMapAdjust;
        RenderTarget = LightMapTex;
    }
    pass
	{
        VertexShader = PostProcessVS;
        PixelShader = PS_DisplayDepth;
        RenderTarget = BufferTex;
	}
    pass
	{
        VertexShader = PostProcessVS;
        PixelShader = PS_NormalBuffer;
        RenderTarget = NormalTex;
	}
    pass
	{
        VertexShader = PostProcessVS;
        PixelShader = PS_ComputeGI;
        RenderTarget = OutputTex;
	}
    pass
	{
        VertexShader = PostProcessVS;
        PixelShader = Output;
	}
}
