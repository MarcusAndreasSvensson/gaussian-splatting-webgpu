struct struct_Uniform {
  data: vec4<u32> 
};

@group(0) @binding(0) var<uniform> tonic: struct_Uniform;
@group(0) @binding(1) var<storage, read_write> input: array<u32>;

@compute @workgroup_size(256)
fn main( 
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let global_idx = global_id.x;
  var tmp: u32;
  let ixj: u32 = global_idx ^ tonic.data.y;

  if ( ixj > global_idx ) {

    if ( ( global_idx & tonic.data.x ) == 0u ) {

      if ( input[ global_idx ] > input[ ixj ] ) {
        tmp = input[ global_idx ];
        input[ global_idx ] = input[ ixj ];
        input[ ixj ] = tmp;
      }
      
    } else {

      if ( input[ global_idx ] < input[ ixj ] ) {
        tmp = input[ global_idx ];
        input[ global_idx ] = input[ ixj ];
        input[ ixj ] = tmp;
      }

    }

  }
}