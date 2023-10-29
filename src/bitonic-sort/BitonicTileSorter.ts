// import { shader1 } from './ComputeShaderCode.wgsl'
import { Renderer } from '@/splatting-web/renderer'
import shader1 from './shader1.wgsl?raw'
import shader2 from './shader2.wgsl?raw'

export class BitonicTileSorter {
  public adapter: GPUAdapter

  public device: GPUDevice

  public maxThreadNum: number = 256

  input: GPUBuffer
  inputTempBuffer: GPUBuffer

  uniform: Uint32Array
  uniformBuffer: GPUBuffer

  pipeline1: GPUComputePipeline
  bindGroup1: GPUBindGroup

  pipeline2: GPUComputePipeline
  bindGroup2: GPUBindGroup

  dispatchIndirectBuffer: GPUBuffer
  dispatchIndirectOffset: number

  numTiles: number
  tileSize: number

  renderer: Renderer
  intersectionOffsetBuffer: GPUBuffer
  intersectionOffsetCountBuffer: GPUBuffer

  constructor(
    adapter: GPUAdapter,
    device: GPUDevice,
    input: GPUBuffer,
    numTiles: number,
    tileSize: number,
    renderer: Renderer,
    intersectionOffsetBuffer: GPUBuffer,
    intersectionOffsetCountBuffer: GPUBuffer,
  ) {
    if (!navigator.gpu) {
      throw new Error('WebGPU not supported!')
    }

    this.adapter = adapter
    this.device = device
    this.renderer = renderer
    this.numTiles = numTiles
    this.input = input
    this.tileSize = tileSize

    this.intersectionOffsetBuffer = intersectionOffsetBuffer
    this.intersectionOffsetCountBuffer = intersectionOffsetCountBuffer

    this.maxThreadNum = this.device.limits.maxComputeWorkgroupSizeX

    this.inputTempBuffer = this.device.createBuffer({
      label: 'inputData',
      size: tileSize * 4,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    })

    this.dispatchIndirectBuffer = this.device.createBuffer({
      label: 'dispatchIndirectBuffer',
      mappedAtCreation: true,
      size: 4 * 3,
      // size: this.numTiles * 4 * 3,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    })

    const test = new Uint32Array(this.dispatchIndirectBuffer.getMappedRange())

    const threadgroupsPerGrid = Math.max(1, this.tileSize / this.maxThreadNum)

    // for (let i = 0; i < this.numTiles; i++) {
    //   test[i * 3 + 0] = threadgroupsPerGrid
    //   test[i * 3 + 1] = 1
    //   test[i * 3 + 2] = 1
    // }

    test[0] = threadgroupsPerGrid
    test[1] = 1
    test[2] = 1

    console.log('test', test)

    this.dispatchIndirectOffset = 0

    this.uniform = new Uint32Array([0, 0, 0, 0])
    // this.uniform = new Uint32Array(Array.from({ length: 4 }, () => 0))

    this.uniformBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    const shaderModule1 = this.device.createShaderModule({
      label: 'shader1',
      code: shader1,
    })

    const bindGroupLayout1 = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'uniform',
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'storage',
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
          },
        },
      ],
    })

    const pipelineLayout1 = this.device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout1],
    })

    this.pipeline1 = this.device.createComputePipeline({
      label: 'pipeline1',

      compute: {
        module: shaderModule1,
        entryPoint: 'main',
      },

      layout: pipelineLayout1,
    })

    this.bindGroup1 = this.device.createBindGroup({
      layout: bindGroupLayout1,

      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.uniformBuffer,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.input,
            // buffer: this.inputTempBuffer,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.intersectionOffsetBuffer,
          },
        },
        {
          binding: 3,
          resource: {
            buffer: this.intersectionOffsetCountBuffer,
          },
        },
      ],
    })

    const shaderModule2 = this.device.createShaderModule({
      label: 'shader2',
      code: shader2,
    })

    const bindGroupLayout2 = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'uniform',
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'storage',
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
          },
        },
      ],
    })

    const pipelineLayout2 = this.device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout2],
    })

    this.pipeline2 = this.device.createComputePipeline({
      label: 'pipeline2',

      compute: {
        module: shaderModule2,
        entryPoint: 'main',
      },

      layout: pipelineLayout2,
    })

    this.bindGroup2 = this.device.createBindGroup({
      layout: bindGroupLayout2,

      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.uniformBuffer,
          },
        },

        {
          binding: 1,
          resource: {
            buffer: this.input,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.intersectionOffsetBuffer,
          },
        },
        {
          binding: 3,
          resource: {
            buffer: this.intersectionOffsetCountBuffer,
          },
        },
      ],
    })
  }

  static async init() {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    })

    if (!adapter) {
      throw new Error('Adapter init failed!')
    }

    const device = await adapter.requestDevice()

    return { adapter, device }
  }

  public async sort(
    numIntersections: number,
    intersectionOffsetCounts: Uint32Array,
  ) {
    const length = 2 ** 9
    // const length = this.input.size / 4 / 2

    console.log('intersectionOffsetCounts', intersectionOffsetCounts)

    console.log('numIntersections sort', numIntersections)

    // // Debug
    // const resultBufferToRead = this.device.createBuffer({
    //   size: this.tileSize * 4,
    //   usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    // })

    const threadgroupsPerGrid = Math.max(1, this.tileSize / this.maxThreadNum)

    console.log('this.maxThreadNum', this.maxThreadNum)

    const offset = Math.max(
      0,
      Math.log2(length) - Math.log2(this.maxThreadNum * 2 + 1),
    )

    console.log('this.input', this.input.size)
    console.log('this.inputTempBuffer', this.inputTempBuffer.size)

    console.log('this.input.size / 4 / 2', this.input.size / 4 / 2)

    const commandEncoder = this.device.createCommandEncoder()

    // Shared memory

    this.uniform[0] = this.input.size / 4 / 2 - numIntersections
    // this.uniform[1] =
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniform)

    const passEncoder = commandEncoder.beginComputePass()
    passEncoder.setPipeline(this.pipeline1)
    passEncoder.setBindGroup(0, this.bindGroup1)
    passEncoder.dispatchWorkgroups(this.numTiles)
    // passEncoder.dispatchWorkgroups(threadgroupsPerGrid)
    // passEncoder.dispatchWorkgroupsIndirect(threadgroupsPerGrid)
    passEncoder.end()

    this.device.queue.submit([commandEncoder.finish()])

    // Shared memory

    console.log('offset', offset)
    console.log('threadgroupsPerGrid', threadgroupsPerGrid)
    console.log('length', length)

    // const indices = []

    // // if (threadgroupsPerGrid > 1) {
    // for (
    //   let subArraySize = threadgroupsPerGrid >> offset;
    //   subArraySize <= length;
    //   subArraySize = subArraySize << 1
    // ) {
    //   indices.push({ subArraySize, compareDist: [] as number[] })

    //   for (
    //     let compareDist = subArraySize >> 1;
    //     compareDist > 0;
    //     compareDist = compareDist >> 1
    //   ) {
    //     indices[indices.length - 1]!.compareDist.push(compareDist)

    //     const commandEncoder = this.device.createCommandEncoder()
    //     const passEncoder = commandEncoder.beginComputePass()

    //     passEncoder.setPipeline(this.pipeline2)
    //     passEncoder.setBindGroup(0, this.bindGroup2)

    //     this.uniform[0] = subArraySize
    //     this.uniform[1] = compareDist
    //     this.uniform[2] = this.input.size / 4 / 2 - numIntersections

    //     this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniform)

    //     // for (let i = 0; i < this.numTiles; i++) {}

    //     // passEncoder.dispatchWorkgroupsIndirect(
    //     //   this.dispatchIndirectBuffer,
    //     //   0,
    //     // )
    //     passEncoder.dispatchWorkgroups(threadgroupsPerGrid)
    //     passEncoder.end()

    //     this.device.queue.submit([commandEncoder.finish()])
    //   }
    // }

    // console.log('indices', indices)

    // }

    {
      const commandEncoder = this.device.createCommandEncoder()

      // // Copy data back to the original buffer.
      // commandEncoder.copyBufferToBuffer(
      //   this.inputTempBuffer,
      //   0,
      //   this.input,
      //   this.input.size - this.inputTempBuffer.size,
      //   this.inputTempBuffer.size,
      // )

      this.renderer.timestamp(commandEncoder, 'bitonic tile sort')

      // // Debug
      // commandEncoder.copyBufferToBuffer(
      //   this.inputTempBuffer,
      //   0,
      //   resultBufferToRead,
      //   0,
      //   this.inputTempBuffer.size,
      // )

      this.device.queue.submit([commandEncoder.finish()])

      // // Debug
      // await resultBufferToRead.mapAsync(GPUMapMode.READ)
      // const result = new Uint32Array(resultBufferToRead.getMappedRange())
      // console.log('GPU', result)

      // let mistakes = 0
      // for (let i = 0; i < result.length; i++) {
      //   if (i !== result.length - 1 && result[i]! > result[i + 1]!) {
      //     mistakes++
      //   }
      // }
      // console.log('mistakes', mistakes)
    }
  }

  public Dispose() {
    this.device?.destroy()
  }
}
