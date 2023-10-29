const workgroup_size = 256u;
const number_per_thread = 1u;
const shared_memory_size = 512u;
// const shared_memory_size = 1024u;
// const number_per_thread = 4u;
// const shared_memory_size = 2048u;

struct Uniform {
  data: vec4<u32> 
};

struct KeyValue {
  key: u32,
  value: u32
};


@group(0) @binding(0) var<uniform> u_params: Uniform;
@group(0) @binding(1) var<storage, read_write> input_data: array<KeyValue>;
@group(0) @binding(2) var<storage, read_write> offset_buffer: array<u32>;
@group(0) @binding(3) var<storage, read_write> offset_buffer_count: array<u32>;


var<workgroup> sharedData: array<KeyValue, shared_memory_size>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
  let local_idx = local_id.x;
  let global_idx = global_id.x;

  let base_offset = u_params.data[0];
  let input_offset = base_offset + offset_buffer[ group_id.x ];
  let input_count = offset_buffer_count[ group_id.x ];

  if ( local_idx < input_count) {
    sharedData[ local_idx ] = input_data[ input_offset + local_idx ];
  }

  workgroupBarrier();
  storageBarrier();

  let offset: u32 = group_id.x * workgroup_size;

  var tmp: KeyValue;

  for ( var k: u32 = 2u; k <= workgroup_size; k = k << 1u ) {
    for ( var j: u32 = k >> 1u; j > 0u; j = j >> 1u ) {

      let ixj: u32 = ( global_idx ^ j ) - offset;

      if ( ixj > local_idx ) {

        if ( ( local_idx & k ) == 0u ) {
          if ( sharedData[ local_idx ].key > sharedData[ ixj ].key ) {
            tmp = sharedData[ local_idx ];
            sharedData[ local_idx ] = sharedData[ ixj ];
            sharedData[ ixj ] = tmp;
          }
        } else {
          if ( sharedData[ local_idx ].key < sharedData[ ixj ].key ) {
            tmp = sharedData[ local_idx ];
            sharedData[ local_idx ] = sharedData[ ixj ];
            sharedData[ ixj ] = tmp;
          }
        }
        
      }

      workgroupBarrier();
      storageBarrier();   
    }
  }

  if ( local_idx < input_count ) {
    let output_offset = workgroup_size - input_count;

    input_data[ input_offset + local_idx ] = sharedData[ output_offset + local_idx ];
  }
}