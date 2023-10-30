const workgroup_size = 256u;
const inputs_per_thread = 8u;
const shared_memory_size = 2048u;
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


  for ( var i: u32 = 0u; i < inputs_per_thread; i++ ) {
    let idx = local_idx * inputs_per_thread + i;

    if ( idx < input_count) {
      sharedData[ idx ] = input_data[ input_offset + idx ];
    }
  }

  workgroupBarrier();
  storageBarrier();


  var tmp: KeyValue;


  for ( var sub_array_size: u32 = 2u; sub_array_size <= shared_memory_size; sub_array_size = sub_array_size << 1u ) {
    for ( var compare_dist: u32 = sub_array_size >> 1u; compare_dist > 0u; compare_dist = compare_dist >> 1u ) {

      for (var i: u32 = 0u; i < inputs_per_thread; i++) {
        let local_idx_i = local_idx * inputs_per_thread + i;
        let ixj: u32 =  (local_idx_i) ^ compare_dist;

        if ( ixj > local_idx_i ) {

          if ( ( local_idx_i & sub_array_size ) == 0u ) {
            if ( sharedData[ local_idx_i ].key > sharedData[ ixj ].key ) {
              tmp = sharedData[ local_idx_i ];
              sharedData[ local_idx_i ] = sharedData[ ixj ];
              sharedData[ ixj ] = tmp;
            }
          } else {
            if ( sharedData[ local_idx_i ].key < sharedData[ ixj ].key ) {
              tmp = sharedData[ local_idx_i ];
              sharedData[ local_idx_i ] = sharedData[ ixj ];
              sharedData[ ixj ] = tmp;
            }
          }
          
        }

        workgroupBarrier();
        storageBarrier();  
      }

    }
  }


  let output_offset = shared_memory_size - input_count;
  

  for (var i: u32 = 0u; i < inputs_per_thread; i++) {
    let idx = local_idx * inputs_per_thread + i;


    if (idx < input_count) {
        input_data[input_offset + idx] = sharedData[output_offset + idx];
    }
  }

}