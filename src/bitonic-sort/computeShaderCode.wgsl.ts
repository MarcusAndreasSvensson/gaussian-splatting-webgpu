export const shader1 = (MAX_THREAD_NUM: number) =>
  `
  struct structFixedData {
    indices: array<u32, ${MAX_THREAD_NUM}>,
    data: array<f32, ${MAX_THREAD_NUM}>
  };

  struct ssboIndices {
    data: array<u32>
  };

  struct ssboData {
    data: array<f32>
  };


  @group(0) @binding(0) var<storage, read_write> inputData: ssboData;
  @group(0) @binding(1) var<storage, read_write> inputIndices: ssboIndices;

  var<workgroup> sharedData: structFixedData;

  @compute @workgroup_size(${MAX_THREAD_NUM}, 1, 1)
  fn main(
      @builtin(local_invocation_id) local_id: vec3<u32>,
      @builtin(global_invocation_id) global_id: vec3<u32>,
      @builtin(workgroup_id) group_id: vec3<u32>,
  ) {
      let localIdX: u32 = local_id.x;
      let globalIdX: u32 = global_id.x;

      sharedData.data[ localIdX ] = inputData.data[ globalIdX ];
      sharedData.indices[ localIdX ] = inputIndices.data[ globalIdX ];

      workgroupBarrier();
      storageBarrier();

      let offset: u32 = group_id.x * ${MAX_THREAD_NUM}u;

      var tmp: f32;
      var tmpIdx: u32;

      for ( var k: u32 = 2u; k <= ${MAX_THREAD_NUM}u; k = k << 1u ) {
        for ( var j: u32 = k >> 1u; j > 0u; j = j >> 1u ) {
          let ixj: u32 = ( globalIdX ^ j ) - offset;

          if ( ixj > localIdX ) {
            if ( ( globalIdX & k ) == 0u ) {
              if ( sharedData.data[ localIdX ] > sharedData.data[ ixj ] ) {
                tmp = sharedData.data[ localIdX ];
                sharedData.data[ localIdX ] = sharedData.data[ ixj ];
                sharedData.data[ ixj ] = tmp;

                tmpIdx = sharedData.indices[ localIdX ];
                sharedData.indices[ localIdX ] = sharedData.indices[ ixj ];
                sharedData.indices[ ixj ] = tmpIdx;
              }
            } else {
              if ( sharedData.data[ localIdX ] < sharedData.data[ ixj ] ) {
                tmp = sharedData.data[ localIdX ];
                sharedData.data[ localIdX ] = sharedData.data[ ixj ];
                sharedData.data[ ixj ] = tmp;

                tmpIdx = sharedData.indices[ localIdX ];
                sharedData.indices[ localIdX ] = sharedData.indices[ ixj ];
                sharedData.indices[ ixj ] = tmpIdx;
              }
            }
          }

          workgroupBarrier();
          storageBarrier();   
        }
      }

      inputData.data[ globalIdX ] = sharedData.data[ localIdX ];
      inputIndices.data[ globalIdX ] = sharedData.indices[ localIdX ];

  }`

export const shader2 = (MAX_THREAD_NUM: number) =>
  `
  struct ssboIndices {
    data: array<u32>
  };

  struct ssbo {
    data: array<f32>
  };

  struct struct_Uniform {
    data: vec4<u32> 
  };

  @group(0) @binding(0)
  var<uniform> tonic: struct_Uniform;

  @group(0) @binding(1) var<storage, read_write> input: ssbo;
  @group(0) @binding(2) var<storage, read_write> inputIndices: ssboIndices;

  @compute @workgroup_size(${MAX_THREAD_NUM},1,1)
  fn main( 
      @builtin(global_invocation_id) global_id: vec3<u32>
  ) {
      let globalIdX: u32 = global_id.x;
      var tmp: f32;
      var tmpIdx: u32;
      let ixj: u32 = globalIdX ^ tonic.data.y;

      if ( ixj > globalIdX ) {
        if ( ( globalIdX & tonic.data.x ) == 0u ) {
          if ( input.data[ globalIdX ] > input.data[ ixj ] ) {
            tmp = input.data[ globalIdX ];
            input.data[ globalIdX ] = input.data[ ixj ];
            input.data[ ixj ] = tmp;

            tmpIdx = inputIndices.data[ globalIdX ];
            inputIndices.data[ globalIdX ] = inputIndices.data[ ixj ];
            inputIndices.data[ ixj ] = tmpIdx;
          }
        } else {
          if ( input.data[ globalIdX ] < input.data[ ixj ] ) {
            tmp = input.data[ globalIdX ];
            input.data[ globalIdX ] = input.data[ ixj ];
            input.data[ ixj ] = tmp;

            tmpIdx = inputIndices.data[ globalIdX ];
            inputIndices.data[ globalIdX ] = inputIndices.data[ ixj ];
            inputIndices.data[ ixj ] = tmpIdx;
          }
        }
      }
    }`
