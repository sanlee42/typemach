{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        rustShell = pkgs.mkShell {
          packages = [
            pkgs.rustup
            pkgs.rust-analyzer
            pkgs.pkg-config
            pkgs.llvmPackages.clang
            pkgs.llvmPackages.libclang
            pkgs.mold
            pkgs.gcc.cc.lib
            pkgs.postgresql
          ];

          shellHook = ''
            if [ -f rust-toolchain.toml ]; then
              rust_version=$(grep 'channel' rust-toolchain.toml | cut -d '"' -f 2)
              rustup override set "$rust_version"
              rustup component add rustfmt --toolchain "$rust_version" 2>/dev/null || true
              rustup component add clippy --toolchain "$rust_version" 2>/dev/null || true
              rustup component add rust-src --toolchain "$rust_version" 2>/dev/null || true
              rustup component add rust-analyzer --toolchain "$rust_version" 2>/dev/null || true
            fi
          '';

          RUSTFLAGS = "-C link-arg=-fuse-ld=mold";
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.gcc.cc.lib ];
        };
      in {
        devShells.default = rustShell;
      });
}
