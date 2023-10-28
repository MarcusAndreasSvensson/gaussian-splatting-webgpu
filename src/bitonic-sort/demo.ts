import { BitonicTileSorter } from './BitonicTileSorter'

export const sortExample = async () => {
  const { adapter, device } = await BitonicTileSorter.init()

  if (!adapter) throw new Error('Adapter init failed!')

  const exponent = 13

  const array = new Uint32Array(Math.pow(2, exponent))

  for (let i = 0; i < array.length; ++i) {
    array[i] = Math.random() * 10000
  }

  const buffer = device.createBuffer({
    size: array.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  })

  const mappedBuffer = new Uint32Array(buffer.getMappedRange())
  mappedBuffer.set(array)

  console.log('input', mappedBuffer)

  buffer.unmap()

  // const sorter = new BitonicTileSorter(adapter, device, buffer, 0, 0)

  // if (await sorter.validate()) {
  //   console.log('GPU sort validation passed!')
  // }
}
