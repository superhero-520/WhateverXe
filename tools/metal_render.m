// Run one language version per process. Requires the macOS 26 SDK.
// xcrun clang -fobjc-arc -framework Foundation -framework Metal tools/metal_render.m -o /tmp/metal_render
// /tmp/metal_render 2.4   or: /tmp/metal_render 3.0 --atomic-float
// --float-via-cas checks a shader fallback. --bindless reads a GPU address via a Tier 2 argument buffer.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef enum { Basic, NativeFloat, FloatViaCAS, Bindless } Probe;

static NSString *const shaderSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "#if PROBE == 3\n"
     "struct Palette { device const uint *red; };\n"
     "#endif\n"
     "vertex float4 triangle(uint i [[vertex_id]]) {\n"
     "    const float2 points[] = {float2(-1,-1), float2(3,-1), float2(-1,3)};\n"
     "    return float4(points[i], 0, 1);\n"
     "}\n"
     "fragment float4 pattern(float4 pos [[position]]\n"
     "#if PROBE == 1\n"
     "        , device atomic_float &count [[buffer(0)]]\n"
     "#elif PROBE == 2\n"
     "        , device atomic_uint &count [[buffer(0)]]\n"
     "#elif PROBE == 3\n"
     "        , constant Palette &palette [[buffer(0)]]\n"
     "#endif\n"
     ") {\n"
     "#if PROBE == 1\n"
     "    atomic_fetch_add_explicit(&count, 0.25f, memory_order_relaxed);\n"
     "#elif PROBE == 2\n"
     "    uint previous = atomic_load_explicit(&count, memory_order_relaxed);\n"
     "    uint next;\n"
     "    do {\n"
     "        next = as_type<uint>(as_type<float>(previous) + 0.25f);\n"
     "    } while (!atomic_compare_exchange_weak_explicit(&count, &previous, next,\n"
     "                     memory_order_relaxed, memory_order_relaxed));\n"
     "#endif\n"
     "    uint red = uint(pos.x);\n"
     "#if PROBE == 3\n"
     "    red = palette.red[red];\n"
     "#endif\n"
     "    return float4(red, uint(pos.y), __METAL_VERSION__ / 10, 255) / 255.0;\n"
     "}\n";

static int failed(const char *stage, NSError *error) {
    fprintf(stderr, "%s failed%s%s\n", stage, error ? ": " : "",
            error ? error.localizedDescription.UTF8String : "");
    return 1;
}

static int draw(id<MTLDevice> device, MTLLanguageVersion version, Probe probe) {
    NSError *error = nil;
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.languageVersion = version;
    options.preprocessorMacros = @{@"PROBE" : @(probe)};
    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:options error:&error];
    if (!library)
        return failed("source compilation", error);

    MTLRenderPipelineDescriptor *pipelineDescriptor = [MTLRenderPipelineDescriptor new];
    pipelineDescriptor.vertexFunction = [library newFunctionWithName:@"triangle"];
    pipelineDescriptor.fragmentFunction = [library newFunctionWithName:@"pattern"];
    pipelineDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    id<MTLRenderPipelineState> pipeline =
        [device newRenderPipelineStateWithDescriptor:pipelineDescriptor error:&error];
    if (!pipeline)
        return failed("render pipeline creation", error);

    enum { width = 16, height = 16, rowBytes = 256 };
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:width height:height mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> target = [device newTextureWithDescriptor:descriptor];
    id<MTLBuffer> result = [device newBufferWithLength:rowBytes * height options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    if (!target || !result || !command)
        return failed("resource allocation", nil);

    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> render = [command renderCommandEncoderWithDescriptor:pass];
    if (!render)
        return failed("render encoder", nil);
    id<MTLBuffer> count = nil;
    if (probe == NativeFloat || probe == FloatViaCAS) {
        const float zero = 0;
        count = [device newBufferWithBytes:&zero length:sizeof(zero) options:MTLResourceStorageModeShared];
        if (!count)
            return failed("counter allocation", nil);
        [render setFragmentBuffer:count offset:0 atIndex:0];
    }
    if (probe == Bindless) {
        if (@available(macOS 13.0, *)) {
            if (device.argumentBuffersSupport != MTLArgumentBuffersTier2)
                return failed("tier 2 argument buffers", nil);
            uint32_t ramp[width];
            for (unsigned x = 0; x < width; ++x)
                ramp[x] = 17 + 3 * x;
            id<MTLBuffer> palette = [device newBufferWithBytes:ramp length:sizeof(ramp)
                                                       options:MTLResourceStorageModeShared];
            if (!palette || ![palette respondsToSelector:@selector(gpuAddress)])
                return failed("GPU buffer address query", nil);
            const uint64_t address = palette.gpuAddress;
            if (!address)
                return failed("GPU buffer address allocation", nil);
            [render useResource:palette usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
            [render setFragmentBytes:&address length:sizeof(address) atIndex:0];
        } else {
            return failed("direct argument buffer encoding on this OS", nil);
        }
    }
    [render setRenderPipelineState:pipeline];
    [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [render endEncoding];

    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (!blit)
        return failed("blit encoder", nil);
    [blit copyFromTexture:target sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(width, height, 1) toBuffer:result destinationOffset:0
    destinationBytesPerRow:rowBytes destinationBytesPerImage:rowBytes * height];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
        return failed("GPU execution", command.error);
    if ((probe == NativeFloat || probe == FloatViaCAS) && *(const float *)count.contents != width * height * 0.25f) {
        fprintf(stderr, "fragment atomic counter: expected %.2f, got %.2f\n",
                width * height * 0.25, *(const float *)count.contents);
        return 1;
    }
    // Blue identifies the compiler's effective language version, independent of the requested option.
    const unsigned requestedLanguage = (version >> 16) * 10 + (version & 0xffff);
    const uint8_t *bytes = result.contents;
    for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
            const unsigned red = probe == Bindless ? 17 + 3 * x : x;
            const uint8_t expected[4] = {red, y, requestedLanguage, 255};
            const uint8_t *pixel = bytes + y * rowBytes + x * 4;
            if (memcmp(pixel, expected, sizeof(expected)) != 0) {
                fprintf(stderr, "pixel (%u,%u): expected %u,%u,%u,255; got %u,%u,%u,%u\n",
                        x, y, red, y, requestedLanguage, pixel[0], pixel[1], pixel[2], pixel[3]);
                return 1;
            }
        }
    }
    printf("Verified %u RGBA pixels through the vertex, fragment and blit stages.\n", width * height);
    if (probe == NativeFloat || probe == FloatViaCAS)
        printf("Verified %u fragment float additions using %s.\n", width * height,
               probe == NativeFloat ? "atomic_float" : "integer compare-and-swap");
    if (probe == Bindless)
        puts("Verified direct Tier 2 argument buffer encoding and indirect fragment reads.");
    return 0;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        Probe probe = Basic;
        if (argc == 3 && !strcmp(argv[2], "--atomic-float"))
            probe = NativeFloat;
        if (argc == 3 && !strcmp(argv[2], "--float-via-cas"))
            probe = FloatViaCAS;
        if (argc == 3 && !strcmp(argv[2], "--bindless"))
            probe = Bindless;
        if ((argc != 2 && probe == Basic) ||
            (strcmp(argv[1], "2.4") && strcmp(argv[1], "3.0") && strcmp(argv[1], "4.0"))) {
            fprintf(stderr, "usage: %s {2.4|3.0|4.0} [--atomic-float|--float-via-cas|--bindless]\n", argv[0]);
            return 2;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device)
            return failed("default Metal device", nil);
        printf("Device: %s; MSL %s\n", device.name.UTF8String, argv[1]);
        if (@available(macOS 13.0, *))
            printf("Metal 3 family: %s\n", [device supportsFamily:MTLGPUFamilyMetal3] ? "yes" : "no");
        if (@available(macOS 26.0, *))
            printf("Metal 4 family: %s\n", [device supportsFamily:MTLGPUFamilyMetal4] ? "yes" : "no");
        fflush(stdout);
        if (!strcmp(argv[1], "2.4")) {
            if (@available(macOS 12.0, *))
                return draw(device, MTLLanguageVersion2_4, probe);
        } else if (!strcmp(argv[1], "3.0")) {
            if (@available(macOS 13.0, *))
                return draw(device, MTLLanguageVersion3_0, probe);
        } else if (@available(macOS 26.0, *)) {
            return draw(device, MTLLanguageVersion4_0, probe);
        }
        return failed("requested MSL version on this OS", nil);
    }
}
