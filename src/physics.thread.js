/**
 * Helper thread for the physics pool.
 *
 * Every helper instantiates the same module over the same shared memory, so the
 * particle arrays are literally the same bytes the host is stepping. Each one
 * needs its own stack and thread locals before it may run any Rust.
 */
self.onmessage = async ({ data: { module, memory, index } }) => {
  const { exports } = await WebAssembly.instantiate(module, { env: { memory } })

  // Stack first: __wasm_init_tls itself runs on it.
  exports.__stack_pointer.value = exports.worker_stack_top(index)
  exports.__wasm_init_tls(exports.worker_tls_base(index))

  // Parks on an atomic wait and never returns.
  exports.worker_run(index)
}
