# Deblur_MCEM
The repo provides the demo code for the nonuniform deblurring method. For more efficient Langevin sampling, we provide the preconditioned gradient descent Langevin (p-LD) iteration, which is used in the demo code.

## Requirement

All the codes are tested on Ubuntu LTS 18.04. Pls ensure the working OS system matches the required OS version.

## Request the demo python file

The following two demo files are accessible upon the request at [matliji@163.com](mailto:matliji@163.com).

```bash
Lai_nonuniform_cvpr_MCEM_pLD.py
Gopro_nonuniform_cvpr_MCEM_pLD.py
```

## Run the demo

### For the Lai nonuniform dataset
    ```bash
    python3 Lai_nonuniform_cvpr_MCEM_pLD.py
    ```

### For the Gopro nonuniform deblurring
    ```bash
    python3 Gopro_nonuniform_cvpr_MCEM_pLD.py
    ```