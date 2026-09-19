from . import install, k


if __name__ == '__main__':
    install()
    from experiments.guidance_dynamic_50k_20260915.worker import main
    try:
        main()
    except k.RequestedStop as error:
        print(str(error), flush=True)
        raise SystemExit(75)
