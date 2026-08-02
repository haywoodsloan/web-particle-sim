import { execFileSync } from 'node:child_process'
import { mkdirSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const require = createRequire(import.meta.url)
const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const target = 'wasm32-unknown-unknown'
const cargo = process.platform === 'win32' ? 'cargo.exe' : 'cargo'
const rustup = process.platform === 'win32' ? 'rustup.exe' : 'rustup'
// Both -Z build-std and stdarch_wasm_atomic_wait are unstable, so an unpinned
// nightly could break the deploy without a commit. Bump this deliberately.
const nightlyVersion = 'nightly-2026-08-02'
// Rebuilding std with atomics needs a host linker, which the MSVC host only has
// with Visual Studio installed; the GNU host ships its own.
const nightly =
  process.env.PHYSICS_NIGHTLY ??
  (process.platform === 'win32'
    ? `${nightlyVersion}-x86_64-pc-windows-gnu`
    : nightlyVersion)

// rust-toolchain.toml can only pin one channel, and an explicitly named one is
// never auto-installed, so the threaded build provisions its own.
function ensureNightly() {
  const installed = execFileSync(rustup, ['toolchain', 'list'], {
    encoding: 'utf8',
  })

  if (!installed.split('\n').some((line) => line.startsWith(nightly))) {
    console.log(`Installing ${nightly} for the threaded module...`)
    execFileSync(
      rustup,
      // std is rebuilt from source, so rust-src is not optional here.
      // prettier-ignore
      [
        'toolchain', 'install', nightly,
        '--component', 'rust-src',
        '--target', target,
        '--profile', 'minimal',
      ],
      { stdio: 'inherit' },
    )
    return
  }

  const components = execFileSync(
    rustup,
    ['component', 'list', '--toolchain', nightly, '--installed'],
    { encoding: 'utf8' },
  )

  if (!components.includes('rust-src')) {
    console.log(`Adding rust-src to ${nightly}...`)
    execFileSync(
      rustup,
      ['component', 'add', 'rust-src', '--toolchain', nightly],
      { stdio: 'inherit' },
    )
  }
}

const threadFlags = [
  '-C target-feature=+simd128,+atomics,+bulk-memory,+mutable-globals',
  '-C link-arg=--shared-memory',
  '-C link-arg=--import-memory',
  '-C link-arg=--max-memory=1073741824',
  '-C link-arg=--export=__stack_pointer',
  '-C link-arg=--export=__tls_base',
  '-C link-arg=--export=__tls_size',
  '-C link-arg=--export=__tls_align',
  '-C link-arg=--export=__wasm_init_tls',
].join(' ')

// Binaryen optimises the finished bytecode, so it reaches whole-module wins
// that LLVM cannot see from Rust's IR. It defaults to MVP, so the features
// rustc already emits have to be named explicitly.
function optimise(input, output, extraFeatures = []) {
  execFileSync(
    process.execPath,
    [
      require.resolve('binaryen/bin/wasm-opt'),
      input,
      '-O3',
      '--enable-simd',
      '--enable-bulk-memory',
      '--enable-bulk-memory-opt',
      '--enable-sign-ext',
      '--enable-mutable-globals',
      '--enable-nontrapping-float-to-int',
      '--enable-multivalue',
      '--enable-reference-types',
      ...extraFeatures,
      '-o',
      output,
    ],
    { stdio: 'inherit' },
  )
}

mkdirSync(join(root, 'src'), { recursive: true })

execFileSync(
  cargo,
  [
    'build',
    '--release',
    '--manifest-path',
    join(root, 'physics/Cargo.toml'),
    '--target',
    target,
  ],
  { stdio: 'inherit' },
)

optimise(
  join(root, 'physics/target', target, 'release/physics.wasm'),
  join(root, 'src/physics.wasm'),
)

// The threaded module imports its memory, so it cannot run without a shared
// one. Visitors who are not cross-origin isolated get the module above instead.
const threadedTargetDir = join(root, 'physics/target-threads')

ensureNightly()

execFileSync(
  cargo,
  [
    `+${nightly}`,
    'build',
    '--release',
    '--manifest-path',
    join(root, 'physics/Cargo.toml'),
    '--target',
    target,
    '-Z',
    'build-std=std,panic_abort',
    '--target-dir',
    threadedTargetDir,
  ],
  { stdio: 'inherit', env: { ...process.env, RUSTFLAGS: threadFlags } },
)

optimise(
  join(threadedTargetDir, target, 'release/physics.wasm'),
  join(root, 'src/physics-threads.wasm'),
  ['--enable-threads'],
)
