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
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let global_idx = global_id.x;

  let base_offset = u_params.data[2];
  let input_offset = base_offset;
  // let input_offset = base_offset + offset_buffer[ group_id.x ];
  // let input_count = offset_buffer_count[ group_id.x ];

  var tmp: KeyValue;
  let ixj: u32 = global_idx ^ u_params.data.y;

  if ( ixj > global_idx ) {

    if ( ( global_idx & u_params.data.x ) == 0u ) {

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