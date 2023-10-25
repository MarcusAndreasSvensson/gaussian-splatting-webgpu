const sh_degree = 3;
const n_sh_coeffs = 16;
const intersection_array_length = 1;

const BLOCK_X = 16;
const BLOCK_Y = 16;

struct GaussData {
  id: i32,
  radii: i32,  // Also signals whether this Gaussian is in frustum.
  depth: f32,
  tiles_touched: u32,
  cum_tiles_touched: u32,
  uv: vec2<f32>,
  conic: vec3<f32>,
  color: vec3<f32>,
  opacity: f32,
}

struct Uniforms {
  viewMatrix: mat4x4<f32>,
  projMatrix: mat4x4<f32>,
  camera_position: vec3<f32>,
  tan_fovx: f32,
  tan_fovy: f32,
  focal_x: f32,
  focal_y: f32,
  scale_modifier: f32,
  screen_size: vec2<u32>,
};

struct AuxData {
  num_intersections: u32,
  num_visible_gaussians: u32,
  min_depth: i32,
  max_depth: i32,
}

struct TileDepthKey {
  key: u32,
  gauss_id: u32,
}


var<workgroup> start_offset_shared: i32;
var<workgroup> end_offset_shared: i32;



@group(0) @binding(0) var color_buffer: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var<storage, read_write> gauss_data: array<GaussData>;
@group(0) @binding(3) var<storage, read_write> intersection_keys: array<TileDepthKey>;
@group(0) @binding(4) var<storage, read_write> intersection_key_offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> auxData: AuxData;


@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(local_invocation_index) local_invocation_index: u32,
) {
  let global_id_x: i32 = i32(global_id.x);
  let global_id_y: i32 = i32(global_id.y);

  let u = f32(global_id.x) / f32(uniforms.screen_size.x);
  let v = f32(global_id.y) / f32(uniforms.screen_size.y);

  let num_tiles_x = u32(uniforms.screen_size.x) / 16u;
  let num_tiles_y = u32(uniforms.screen_size.y) / 16u;

  let tile_index_x = workgroup_id.x;
  let tile_index_y = workgroup_id.y;

  let tile_index = u32(tile_index_x + tile_index_y * num_tiles_x);


  let num_groups = uniforms.screen_size.x / 16u * uniforms.screen_size.y / 16u;

  let tile_id = i32(
    workgroup_id.x +
    workgroup_id.y * num_workgroups.x +
    workgroup_id.z * num_workgroups.x * num_workgroups.y
  );

  let thread_idx = i32(local_invocation_index);


  let value = intersection_key_offsets[tile_id];
  var max_value = 0u;

  for (var i = 0u; i < num_groups; i++) {
    max_value = max(max_value, intersection_key_offsets[i]);
  }


  textureStore(
    color_buffer,
    vec2<i32>(global_id_x, global_id_y),
    vec4<f32>(0.0, 0.0, 0.0, 1.0)
  );



  let base_offset = u32(intersection_array_length) - auxData.num_intersections;
  
  if(thread_idx == 0) {
    
    var start_offset = 0u;
    var end_offset = 0u;

    for (var i = 0; i <= tile_id; i++) {
      end_offset += intersection_key_offsets[i];
      start_offset = end_offset - intersection_key_offsets[i];
    }

    start_offset_shared = i32(start_offset);
    end_offset_shared = i32(end_offset);
  }

  var accumulated_color: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
  var accumulated_opacity: f32 = 0.0;
  
  var t_i: f32 = 1.0; // The initial value of accumulated alpha (initial value of accumulated multiplication)

  
  workgroupBarrier();

  let start_offset_local = workgroupUniformLoad(&start_offset_shared);
  let end_offset_local = workgroupUniformLoad(&end_offset_shared);


  for (var idx = start_offset_local; idx < end_offset_local; idx++) {
    
    workgroupBarrier();

    let i = base_offset + u32(idx);

    let gauss_id = intersection_keys[i].gauss_id;

    let conic = gauss_data[gauss_id].conic;
    let opacity = gauss_data[gauss_id].opacity;
    let xy = vec2(
      gauss_data[gauss_id].uv.x * f32(uniforms.screen_size.x),
      gauss_data[gauss_id].uv.y * f32(uniforms.screen_size.y),
    );

    let xy_pixel = vec2(
      u * f32(uniforms.screen_size.x),
      v * f32(uniforms.screen_size.y)
    );

    var distance = vec2(
      xy.x - xy_pixel.x,
      xy.y - xy_pixel.y
    );

    let power = -0.5 *
      (conic.x * distance.x * distance.x + conic.z * distance.y * distance.y) -
      conic.y * distance.x * distance.y;


    let alpha = min(0.99, opacity * exp(power));

    let test_t = t_i * (1.0 - alpha);

    if(
      power <= 0.0f && 
      alpha >= 1.0 / 255.0 && 
      test_t >= 0.0001f
    ) {
      accumulated_color += gauss_data[gauss_id].color * alpha * t_i;

      t_i = test_t;
    }

  }


  textureStore(
    color_buffer,
    vec2<i32>(global_id_x, global_id_y),
    vec4<f32>(accumulated_color, 1.0)
  );
}
