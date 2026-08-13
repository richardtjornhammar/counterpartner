{ pkgs ? import <nixpkgs> {} }:
  pkgs.mkShell {
    buildInputs = [ pkgs.python312 
                    pkgs.python312Packages.pandas 
                    pkgs.python312Packages.numpy 
                    pkgs.python312Packages.scipy
                    pkgs.python312Packages.matplotlib ];
}
