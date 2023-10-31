const sorting_threshold = 2048u;
const workgroup_size = 256u;
// const sorting_threshold = 4096u;

struct Uniform {
  data: vec4<u32> 
};

struct KeyValue {
  key: u32,
  value: u32
};

@group(0) @binding(0) var<uniform> u_params: Uniform;
@group(0) @binding(1) var<storage, read_write> input: array<KeyValue>;
@group(0) @binding(2) var<storage, read_write> offset_buffer: array<u32>;
@group(0) @binding(3) var<storage, read_write> offset_buffer_count: array<u32>;

@compute @workgroup_size(256)
fn main( 
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_groups: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  // let group_id_flat = group_id.x * num_groups.y + group_id.y;
  let group_id_flat = group_id.x + group_id.y * num_groups.x;
  
  let input_count = offset_buffer_count[ group_id_flat ];
  // let input_count = offset_buffer_count[ group_id.x ];

  if (input_count < sorting_threshold) {
    return;
  }

  let group_depth_offset = group_id.z * workgroup_size;

  if (group_depth_offset + local_id.x >= input_count) {
    return;
  }

  let base_offset = u_params.data[0];
  let input_offset = base_offset + offset_buffer[ group_id_flat ];

  // let input_offset = base_offset + offset_buffer[ group_id.x ];


  let global_idx = input_offset + group_depth_offset + local_id.x;

  // input[ global_idx ] = KeyValue( 0u, 0u );

  var tmp: KeyValue;
  let ixj: u32 = global_idx ^ u_params.data[2];

  // let i = index & (size - 1u);
  // let j = index ^ stride;


  // if ( ixj > global_idx ) {
  // if ( ixj > global_idx && ixj < input_offset + input_count ) {
  if (ixj > global_idx && 
    global_idx < input_offset + input_count && 
    ixj < input_offset + input_count) {
  // if (j > index && index < length && j < length) {

    

    if ( ( global_idx & u_params.data[1] ) == 0u ) {
    // if ( ( global_idx & u_params.data[1] ) == 0u && ixj < input_offset + input_count  ) {

      // if(ixj < input_offset + input_count) {
        
      // }


      if ( input[ global_idx ].key > input[ ixj ].key ) {
        tmp = input[ global_idx ];
        input[ global_idx ] = input[ ixj ];
        input[ ixj ] = tmp;
      }
      
    } else {

      if ( input[ global_idx ].key < input[ ixj ].key ) {
        tmp = input[ global_idx ];
        input[ global_idx ] = input[ ixj ];
        input[ ixj ] = tmp;
      }

    }

  }
}
