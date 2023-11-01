import { RadixSort } from '@/radix-sort/radixSortApple'
import {
  StaticArray,
  Struct,
  f32,
  i32,
  u32,
  vec2,
  vec3,
} from '@/splatting-web/packing'
import { PackedGaussians } from '@/splatting-web/ply'

import { GpuContext } from '@/point-preprocessor/gpuContext'
import { Renderer } from '@/splatting-web/renderer'
import gaussShader from './gauss.wgsl?raw'
import intersectionOffsetPrefixSumShader from './intersectionOffsetprefixSum.wgsl?raw'

import { BitonicTileSorter } from '@/bitonic-sort/BitonicTileSorter'
import tileDepthShader from './tileDepthKey.wgsl?raw'

function nextPowerOfTwo(x: number) {
  return Math.pow(2, Math.ceil(Math.log2(x)))
}

const resultLayout = new Struct([
  ['id', i32],
  ['radii', i32],
  ['depth', f32],
  ['uv', new vec2(f32)],
  ['conic', new vec3(f32)],
  ['color', new vec3(f32)],
  ['opacity', f32],
])

const auxLayout = new Struct([
  ['num_intersections', u32],
  ['num_visible_gaussians', u32],
  ['min_depth', i32],
  ['max_depth', i32],
])

const prefixLayout = u32

export const tileDepthKeyLayout = new Struct([
  ['key', u32],
  ['gauss_id', u32],
])

// const numTilesArray = nextPowerOfTwo(1 * 1e6)
// const numTilesArray = 3 * 1e6
// const numTilesArray = nextPowerOfTwo(3 * 1e6)
// const numTilesArray = 6 * 1e6
// const numTilesArray = nextPowerOfTwo(6 * 1e6)
const numTilesArray = 14 * 1e6
// const numTilesArray = 16 * 1e6

export const tileDepthKeyArrayLayoutTest = new StaticArray(
  tileDepthKeyLayout,
  numTilesArray,
)

export class Preprocessor {
  context: GpuContext

  gaussians: PackedGaussians
  numGaussians: number
  pointDataBuffer: GPUBuffer
  numPadded: number

  numThreads: number

  pipelineLayout: GPUPipelineLayout
  pipeline: GPUComputePipeline

  tileDepthKeyPipeline: GPUComputePipeline
  intersectionOffsetPrefixSumPipeline: GPUComputePipeline
  // @ts-ignore
  bindGroup: GPUBindGroup

  resultBuffer: GPUBuffer
  uniformBuffer: GPUBuffer
  tileDepthKeyBuffer: GPUBuffer

  auxBuffer: GPUBuffer
  auxBufferRead: GPUBuffer
  intersectionOffsetBuffer: GPUBuffer
  intersectionOffsetCountBuffer: GPUBuffer
  emptyPrefixBuffer: GPUBuffer
  prefixBuffer: GPUBuffer

  public resultArrayLayout: StaticArray
  numIntersections = 0
  public tileDepthKeyArrayLayout: StaticArray
  public prefixArrayLayout: StaticArray
  public intersectionOffsetArrayLayout: StaticArray
  public cumSumTiles: number[] = []

  maxBufferSize: number
  numTiles: number

  radixSorter: RadixSort
  bitonicTileSorter: BitonicTileSorter

  renderer: Renderer

  constructor(
    context: GpuContext,
    gaussians: PackedGaussians,
    pointDataBuffer: GPUBuffer,
    uniformBuffer: GPUBuffer,
    canvas: HTMLCanvasElement,
    renderer: Renderer,
  ) {
    this.context = context
    console.log('Device limits:', this.context.device.limits)

    this.maxBufferSize = this.context.device.limits.maxBufferSize

    this.renderer = renderer

    this.gaussians = gaussians
    this.numGaussians = gaussians.numGaussians
    this.numPadded = nextPowerOfTwo(gaussians.numGaussians)

    console.log('this.numGaussians', this.numGaussians)

    this.pointDataBuffer = pointDataBuffer
    this.uniformBuffer = uniformBuffer

    this.numThreads = 2048

    const numtilesX = Math.ceil(canvas.width / 16)
    const numtilesY = Math.ceil(canvas.height / 16)
    this.numTiles = numtilesX * numtilesY

    // const itemsPerThread = Math.ceil(this.numGaussians / this.numThreads)

    this.resultArrayLayout = new StaticArray(resultLayout, this.numGaussians)
    this.tileDepthKeyArrayLayout = new StaticArray(
      tileDepthKeyLayout,
      numTilesArray,
    )

    this.prefixArrayLayout = new StaticArray(prefixLayout, this.numThreads)

    this.intersectionOffsetArrayLayout = new StaticArray(
      u32,
      this.numTiles * renderer.numTileDepthSections,
    )

    this.resultBuffer = this.context.device.createBuffer({
      size: this.resultArrayLayout.size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
      mappedAtCreation: false,
      label: 'preprocessor.resultBuffer',
    })

    this.auxBuffer = this.context.device.createBuffer({
      size: auxLayout.size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
      mappedAtCreation: false,
      label: 'preprocessor.auxBuffer',
    })

    this.auxBufferRead = this.context.device.createBuffer({
      size: auxLayout.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      mappedAtCreation: false,
      label: 'preprocessor.auxBufferRead',
    })

    this.intersectionOffsetBuffer = this.context.device.createBuffer({
      size: this.intersectionOffsetArrayLayout.size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
      mappedAtCreation: false,
      label: 'preprocessor.intersectionOffsetBuffer',
    })

    this.intersectionOffsetCountBuffer = this.context.device.createBuffer({
      size: this.intersectionOffsetArrayLayout.size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
      mappedAtCreation: false,
      label: 'preprocessor.intersectionOffsetCountBuffer',
    })

    this.emptyPrefixBuffer = this.context.device.createBuffer({
      size: this.prefixArrayLayout.size,
      usage: GPUBufferUsage.COPY_SRC,
      mappedAtCreation: false,
    })

    this.prefixBuffer = this.context.device.createBuffer({
      size: this.prefixArrayLayout.size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
      mappedAtCreation: false,
      label: 'preprocessor.prefixBuffer',
    })

    this.tileDepthKeyBuffer = this.context.device.createBuffer({
      size: this.tileDepthKeyArrayLayout.size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
      mappedAtCreation: false,
      label: 'preprocessor.tileDepthKeyBuffer',
    })

    this.radixSorter = new RadixSort(
      context,
      this.tileDepthKeyArrayLayout.nElements,
      2,
      this.renderer,
    )

    this.bitonicTileSorter = new BitonicTileSorter(
      context.adapter,
      context.device,
      this.tileDepthKeyBuffer,
      numtilesX,
      numtilesY,
      2 ** 8,
      this.renderer,
      this.intersectionOffsetBuffer,
      this.intersectionOffsetCountBuffer,
    )

    const computeBindGroupLayout = this.createUniforms()

    this.pipelineLayout = this.context.device.createPipelineLayout({
      bindGroupLayouts: [computeBindGroupLayout],
    })

    const gaussShaderWithParams = gaussShader
      .replace(
        'const num_gauss_unpaddded: i32 = 1;',
        `const num_gauss_unpaddded: i32 = ${this.numGaussians};`,
      )
      .replace(
        'const num_depth_tiles = 2u;',
        `const num_depth_tiles = ${this.renderer.numTileDepthSections}u;`,
      )

    this.pipeline = this.context.device.createComputePipeline({
      layout: this.pipelineLayout,
      compute: {
        module: this.context.device.createShaderModule({
          code: gaussShaderWithParams,
        }),
        entryPoint: 'main',
        constants: {
          // num_gauss_unpaddded: this.numGaussians,
        },
      },
    })

    const intersectionOffsetPrefixSumShaderWithParams =
      intersectionOffsetPrefixSumShader
        .replace(
          'const num_tiles: i32 = 1;',
          `const num_tiles: i32 = ${this.numTiles};`,
        )
        .replace(
          'const num_depth_tiles = 2;',
          `const num_depth_tiles = ${this.renderer.numTileDepthSections};`,
        )

    this.intersectionOffsetPrefixSumPipeline =
      this.context.device.createComputePipeline({
        layout: this.pipelineLayout,
        compute: {
          module: this.context.device.createShaderModule({
            code: intersectionOffsetPrefixSumShaderWithParams,
          }),
          entryPoint: 'main',
          constants: {
            // num_gauss_unpaddded: this.numGaussians,
          },
        },
      })

    const tileDepthShaderWithParams = tileDepthShader
      .replace(
        'const num_gauss_unpaddded: i32 = 1;',
        `const num_gauss_unpaddded: i32 = ${this.numGaussians};`,
      )
      .replace(
        'const intersection_array_length = 1;',
        `const intersection_array_length = ${
          this.radixSorter.buf_data.size / (4 * 2)
        };`,
      )
      .replace(
        'const num_depth_tiles = 2u;',
        `const num_depth_tiles = ${this.renderer.numTileDepthSections}u;`,
      )

    this.tileDepthKeyPipeline = this.context.device.createComputePipeline({
      layout: this.pipelineLayout,
      compute: {
        module: this.context.device.createShaderModule({
          code: tileDepthShaderWithParams,
        }),
        entryPoint: 'main',
      },
    })
  }

  private createUniforms() {
    const computeBindGroupLayout = this.context.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'read-only-storage',
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'uniform',
          },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'storage',
          },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'storage',
            // hasDynamicOffset: true,
          },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'storage',
          },
        },
        {
          binding: 5,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'storage',
          },
        },
        {
          binding: 6,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'storage',
          },
        },
        {
          binding: 7,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'storage',
          },
        },
      ],
    })

    this.bindGroup = this.context.device.createBindGroup({
      layout: computeBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.pointDataBuffer,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.uniformBuffer,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.resultBuffer,
          },
        },
        {
          binding: 3,
          resource: {
            buffer: this.tileDepthKeyBuffer,
          },
        },
        {
          binding: 4,
          resource: {
            buffer: this.prefixBuffer,
          },
        },
        {
          binding: 5,
          resource: {
            buffer: this.auxBuffer,
          },
        },
        {
          binding: 6,
          resource: {
            buffer: this.intersectionOffsetBuffer,
          },
        },
        {
          binding: 7,
          resource: {
            buffer: this.intersectionOffsetCountBuffer,
          },
        },
      ],
    })

    return computeBindGroupLayout
  }

  public destroy() {}

  async run() {
    // Debug
    const intersectionOffsetToRead = this.context.device.createBuffer({
      size: this.intersectionOffsetArrayLayout.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      mappedAtCreation: false,
      label: 'preprocessor.intersectionOffsetToRead',
    })

    // Debug
    const intersectionOffsetCountToRead = this.context.device.createBuffer({
      size: this.intersectionOffsetArrayLayout.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      mappedAtCreation: false,
      label: 'preprocessor.intersectionOffsetCountToRead',
    })

    // // Debug
    // const tileDepthKeyToRead = this.context.device.createBuffer({
    //   size: this.tileDepthKeyArrayLayout.size,
    //   usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    //   mappedAtCreation: false,
    //   label: 'preprocessor.tileDepthKeyToRead',
    // })

    // Debug gaussians
    const gaussiansToRead = this.context.device.createBuffer({
      size: this.resultBuffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      mappedAtCreation: false,
      label: 'preprocessor.gaussiansToRead',
    })

    const commandEncoder = this.context.device.createCommandEncoder()

    this.renderer.timestamp(commandEncoder, 'preprocess start')

    commandEncoder.clearBuffer(this.resultBuffer)
    commandEncoder.clearBuffer(this.tileDepthKeyBuffer)
    commandEncoder.clearBuffer(this.prefixBuffer)
    commandEncoder.clearBuffer(this.auxBuffer)
    commandEncoder.clearBuffer(this.intersectionOffsetBuffer)
    commandEncoder.clearBuffer(this.intersectionOffsetCountBuffer)

    this.renderer.timestamp(commandEncoder, 'clear')

    const gaussEncoder = commandEncoder.beginComputePass()
    gaussEncoder.setPipeline(this.pipeline)
    gaussEncoder.setBindGroup(0, this.bindGroup)
    gaussEncoder.dispatchWorkgroups(Math.ceil(this.numGaussians / 256) + 1)
    gaussEncoder.end()

    this.renderer.timestamp(commandEncoder, 'gauss properties')

    const intersectionOffsetPrefixSumEncoder = commandEncoder.beginComputePass()
    intersectionOffsetPrefixSumEncoder.setPipeline(
      this.intersectionOffsetPrefixSumPipeline,
    )
    intersectionOffsetPrefixSumEncoder.setBindGroup(0, this.bindGroup)
    intersectionOffsetPrefixSumEncoder.dispatchWorkgroups(1)
    intersectionOffsetPrefixSumEncoder.end()

    this.renderer.timestamp(commandEncoder, 'intersection offset prefix sum')

    const tileDepthKeyPassEncoder = commandEncoder.beginComputePass()
    tileDepthKeyPassEncoder.setPipeline(this.tileDepthKeyPipeline)
    tileDepthKeyPassEncoder.setBindGroup(0, this.bindGroup)
    // tileDepthKeyPassEncoder.dispatchWorkgroups(this.numTiles)
    tileDepthKeyPassEncoder.dispatchWorkgroups(this.numGaussians / 256 + 1)
    tileDepthKeyPassEncoder.end()

    this.renderer.timestamp(commandEncoder, 'tile depth key')

    commandEncoder.copyBufferToBuffer(
      this.auxBuffer,
      0,
      this.auxBufferRead,
      0,
      auxLayout.size,
    )

    // Debug
    commandEncoder.copyBufferToBuffer(
      this.intersectionOffsetBuffer,
      0,
      intersectionOffsetToRead,
      0,
      this.intersectionOffsetArrayLayout.size,
    )

    // Debug
    commandEncoder.copyBufferToBuffer(
      this.intersectionOffsetCountBuffer,
      0,
      intersectionOffsetCountToRead,
      0,
      this.intersectionOffsetArrayLayout.size,
    )

    // // Debug
    // commandEncoder.copyBufferToBuffer(
    //   this.tileDepthKeyBuffer,
    //   0,
    //   tileDepthKeyToRead,
    //   0,
    //   this.tileDepthKeyArrayLayout.size,
    // )

    // Debug
    commandEncoder.copyBufferToBuffer(
      this.resultBuffer,
      0,
      gaussiansToRead,
      0,
      this.resultArrayLayout.size,
    )

    // NOT DEBUG
    this.context.device.queue.submit([commandEncoder.finish()])

    // Debug
    await intersectionOffsetToRead.mapAsync(
      GPUMapMode.READ,
      0,
      this.intersectionOffsetArrayLayout.size,
    )
    const intersectionOffsets = new Uint32Array(
      intersectionOffsetToRead.getMappedRange(
        0,
        this.intersectionOffsetArrayLayout.size,
      ),
    )
    console.log('intersectionOffsets', intersectionOffsets)
    // console.log(
    //   'intersectionOffsets',
    //   intersectionOffsets.at(-3),
    //   intersectionOffsets.at(-2),
    //   intersectionOffsets.at(-1),
    // )
    // console.log(
    //   'intersectionOffsets',
    //   intersectionOffsets.at(-2)! - intersectionOffsets.at(-3)!,
    //   intersectionOffsets.at(-1)! - intersectionOffsets.at(-2)!,
    // )

    // // intersectionOffsetToRead.unmap()

    // Debug
    await intersectionOffsetCountToRead.mapAsync(
      GPUMapMode.READ,
      0,
      this.intersectionOffsetArrayLayout.size,
    )
    const intersectionOffsetsCount = new Uint32Array(
      intersectionOffsetCountToRead.getMappedRange(
        0,
        this.intersectionOffsetArrayLayout.size,
      ),
    )
    console.log('intersectionOffsetsCount', intersectionOffsetsCount)
    // console.log(
    //   'max',
    //   Math.max(
    //     ...intersectionOffsetsCount.slice(
    //       0,
    //       intersectionOffsetsCount.length - 1,
    //     ),
    //   ),
    // )

    // console.log(
    //   'intersectionOffsetsCount',
    //   intersectionOffsetsCount.at(-3),
    //   intersectionOffsetsCount.at(-2),
    //   intersectionOffsetsCount.at(-1),
    // )
    // // intersectionOffsetCountToRead.unmap()

    // // Debug
    // await gaussiansToRead.mapAsync(
    //   GPUMapMode.READ,
    //   0,
    //   this.resultArrayLayout.size,
    // )
    // const gaussians = this.resultArrayLayout.unpack(
    //   0,
    //   new DataView(
    //     // gaussiansToRead.getMappedRange(0, resultLayout.size * 10000),
    //     gaussiansToRead.getMappedRange(0, this.resultArrayLayout.size),
    //   ),
    // )[1]!
    // const depths = []
    // for (let i = 0; i < 5000; i++) {
    //   // for (let i = 0; i < gaussians.length; i++) {
    //   depths.push(gaussians[i]!.depth)
    // }
    // // const depths = gaussians.map((g) => g.depth)
    // // console.log('gaussians', gaussians)
    // console.log('depths', depths)

    // NOT DEBUG
    await this.auxBufferRead.mapAsync(GPUMapMode.READ, 0, 4)
    this.numIntersections = new Uint32Array(
      this.auxBufferRead.getMappedRange(0, 4),
    )[0]!

    console.log('numIntersections', this.numIntersections)

    // // Debug
    // await tileDepthKeyToRead.mapAsync(
    //   GPUMapMode.READ,
    //   0,
    //   this.tileDepthKeyArrayLayout.size,
    // )
    // const tileDepthKey = new Uint32Array(
    //   tileDepthKeyToRead.getMappedRange(
    //     this.tileDepthKeyArrayLayout.size - this.numIntersections * 2 * 4,
    //     this.numIntersections * 2 * 4,
    //   ),
    //   // tileDepthKeyToRead.getMappedRange(0, this.tileDepthKeyArrayLayout.size),
    // )

    // console.log('tileDepthKey', tileDepthKey)

    // const condition = (count: number) => count < 2 ** 12 && count > 2 ** 11
    // // const condition = (count: number) => count < 2 ** 11 && count > 2 ** 10
    // // const condition = (count: number) => count < 2 ** 9 && count > 2 ** 8

    // for (let i = 0; i < intersectionOffsetsCount.length; i++) {
    //   const count = intersectionOffsetsCount[i]!
    //   const offset = intersectionOffsets[i]!

    //   if (condition(count)) {
    //     // if (count < 256) {
    //     console.log('count', i, count, offset)

    //     const tile = tileDepthKey.slice(offset * 2, (offset + count) * 2 + 2)

    //     const tileKeys = tile.filter((_, i) => i % 2 === 0)

    //     console.log('tile', tile)
    //     console.log('tileKeys', tileKeys)

    //     let mistakes = 0
    //     for (let i = 1; i < tileKeys.length; i++) {
    //       const prevKey = tileKeys[i - 1]
    //       const key = tileKeys[i]

    //       if (key! < prevKey!) {
    //         mistakes++
    //       }
    //     }

    //     console.log('mistakes', mistakes)

    //     break
    //   }
    // }

    // tileDepthKeyToRead.unmap()

    // const mistakes = []
    // for (let i = 1; i < tileDepthKey.length; i += 2) {
    //   const prevKey = tileDepthKey[i * 2 - 2]

    //   const key = tileDepthKey[i * 2]

    //   if (key! < prevKey!) {
    //     mistakes.push([i, prevKey, key])
    //   }
    // }

    // console.log('mistakes', mistakes.length, mistakes)
    // console.log('tileDepthKey', tileDepthKey)

    // // const { values: sortedValuesRadix } =
    // this.radixSorter.sort(
    //   this.tileDepthKeyBuffer,
    //   numTilesArray,
    //   // this.numIntersections,
    // )

    // NOT DEBUG
    this.auxBufferRead.unmap()

    // await this.bitonicTileSorter.sort(
    //   this.numIntersections,
    //   // intersectionOffsetsCount,
    // )

    // {
    //   const commandEncoder = this.context.device.createCommandEncoder()

    //   // Debug
    //   commandEncoder.copyBufferToBuffer(
    //     this.tileDepthKeyBuffer,
    //     0,
    //     tileDepthKeyToRead,
    //     0,
    //     this.tileDepthKeyArrayLayout.size,
    //   )

    //   this.context.device.queue.submit([commandEncoder.finish()])

    //   // Debug
    //   await tileDepthKeyToRead.mapAsync(
    //     GPUMapMode.READ,
    //     0,
    //     this.tileDepthKeyArrayLayout.size,
    //   )
    //   const tileDepthKeySorted = new Uint32Array(
    //     tileDepthKeyToRead.getMappedRange(
    //       this.tileDepthKeyArrayLayout.size - this.numIntersections * 2 * 4,
    //       this.numIntersections * 2 * 4,
    //     ),
    //     // tileDepthKeyToRead.getMappedRange(0, this.tileDepthKeyArrayLayout.size),
    //   )

    //   // console.log('tileDepthKeySorted', tileDepthKeySorted)

    //   for (let i = 0; i < intersectionOffsetsCount.length; i++) {
    //     const count = intersectionOffsetsCount[i]!
    //     const offset = intersectionOffsets[i]!

    //     if (condition(count)) {
    //       // if (count < 256) {
    //       console.log('count', i, count, offset)

    //       const tile = tileDepthKeySorted.slice(
    //         offset * 2,
    //         (offset + count) * 2 + 2,
    //       )

    //       const tileKeys = tile.filter((_, i) => i % 2 === 0)

    //       console.log('tile', tile)
    //       console.log('tileKeys', tileKeys)

    //       const mistakes = []
    //       for (let i = 1; i < tileKeys.length; i++) {
    //         const prevKey = tileKeys[i - 1]
    //         const key = tileKeys[i]

    //         if (key! < prevKey!) {
    //           mistakes.push([i, prevKey, key])
    //           // console.log('mistake', i, prevKey, key)
    //         }
    //       }

    //       console.log('mistakes', mistakes)

    //       break
    //     }
    //   }
    // }
  }
}
