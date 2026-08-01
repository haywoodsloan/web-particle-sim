import { execFileSync } from 'node:child_process'
import { mkdirSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const require = createRequire(import.meta.url)
const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const target = 'wasm32-unknown-unknown'
const compiled = join(
  root,
  'physics/target',
  target,
  'release/physics.wasm',
)
const bundled = join(root, 'src/physics.wasm')

execFileSync(
  process.platform === 'win32' ? 'cargo.exe' : 'cargo',
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

mkdirSync(join(root, 'src'), { recursive: true })
// Binaryen optimises the finished bytecode, so it reaches whole-module wins
// that LLVM cannot see from Rust's IR. It defaults to MVP, so the features
// rustc already emits have to be named explicitly.
execFileSync(
  process.execPath,
  [
    require.resolve('binaryen/bin/wasm-opt'),
    compiled,
    '-O3',
    '--enable-simd',
    '--enable-bulk-memory',
    '--enable-bulk-memory-opt',
    '--enable-sign-ext',
    '--enable-mutable-globals',
    '--enable-nontrapping-float-to-int',
    '--enable-multivalue',
    '--enable-reference-types',
    '-o',
    bundled,
  ],
  { stdio: 'inherit' },
)
