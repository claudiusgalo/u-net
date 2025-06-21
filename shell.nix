{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python3_13;

  # Override torch and torchvision to use same torch derivation
  torch = python.pkgs.pytorch.override {
    cudaSupport = true;
  };

  torchvision = python.pkgs.torchvision.override {
    inherit torch;
  };

  python-with-ml = pkgs.python3.withPackages (ps: with ps; [
    numpy
    scipy
    matplotlib
    pandas
    ipython
    jupyter
    # Use PyTorch with CUDA support
    (ps.pytorch.override {
      cudaSupport = true;
    })
    #torchvision
    #torchaudio
  ]);
in
pkgs.mkShell {
  name = "unet-dev-env";

  buildInputs = [
    python-with-ml
    pkgs.cudaPackages.cudatoolkit
    pkgs.cudaPackages.cudnn
  ];

  shellHook = ''
    export PYTHONPATH=./:$PYTHONPATH
    export CUDA_VISIBLE_DEVICES=0
    echo "✅ U-Net CUDA environment is ready"
  '';
}

