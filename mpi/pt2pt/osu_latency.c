#define BENCHMARK "OSU MPI%s Latency Test"
/*
 * Copyright (C) 2002-2020 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University. 
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
#include <osu_util_mpi.h>

int
main (int argc, char *argv[])
{
    int myid, numprocs, i;
    int size;
    MPI_Status reqstat;
    char *s_buf, *r_buf;
    char *r_gpubuf, *s_gpubuf;
    char *tr_gpubuf, *ts_gpubuf;
    char *r_cpubuf, *s_cpubuf;
    int is_hh = 0, is_dd = 0;

    double t_start = 0.0, t_end = 0.0;
    int po_ret = 0;
    options.bench = PT2PT;
    options.subtype = LAT;

    set_header(HEADER);
    set_benchmark_name("osu_latency");

    po_ret = process_options(argc, argv);

    if (PO_OKAY == po_ret && NONE != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myid));

    if (0 == myid) {
        switch (po_ret) {
            case PO_CUDA_NOT_AVAIL:
                fprintf(stderr, "CUDA support not enabled.  Please recompile "
                        "benchmark with CUDA support.\n");
                break;
            case PO_OPENACC_NOT_AVAIL:
                fprintf(stderr, "OPENACC support not enabled.  Please "
                        "recompile benchmark with OPENACC support.\n");
                break;
            case PO_BAD_USAGE:
                print_bad_usage_message(myid);
                break;
            case PO_HELP_MESSAGE:
                print_help_message(myid);
                break;
            case PO_VERSION_MESSAGE:
                print_version_message(myid);
                MPI_CHECK(MPI_Finalize());
                exit(EXIT_SUCCESS);
            case PO_OKAY:
                break;
        }
    }

    switch (po_ret) {
        case PO_CUDA_NOT_AVAIL:
        case PO_OPENACC_NOT_AVAIL:
        case PO_BAD_USAGE:
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_FAILURE);
        case PO_HELP_MESSAGE:
        case PO_VERSION_MESSAGE:
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_SUCCESS);
        case PO_OKAY:
            break;
    }

    if(numprocs != 2) {
        if(myid == 0) {
            fprintf(stderr, "This test requires exactly two processes\n");
        }

        MPI_CHECK(MPI_Finalize());
        exit(EXIT_FAILURE);
    }

    if (allocate_memory_pt2pt(&s_buf, &r_buf, myid)) {
        /* Error allocating memory */
        MPI_CHECK(MPI_Finalize());
        exit(EXIT_FAILURE);
    }

    if (options.cpy_from_d) {
        if (cudaMalloc((void **) &r_gpubuf, options.max_message_size)) {
            fprintf(stderr, "Error allocating receiving GPU memory %lu\n",
                    options.max_message_size);
            return 1;
        }
        if (cudaMalloc((void **) &s_gpubuf, options.max_message_size)) {
            fprintf(stderr, "Error allocating sending GPU memory %lu\n",
                    options.max_message_size);
            return 1;
        }

        if (options.src == 'H' && options.dst == 'H') {
            if (myid == 0) {
                printf("** Host to Host\n");
            }
            is_hh = 1;
            if (options.add_serial) {
                printf("SDH HDS mode is enabled\n");
                if (cudaMalloc((void **) &tr_gpubuf, options.max_message_size)) {
                    fprintf(stderr, "Error allocating receiving tGPU memory %lu\n",
                            options.max_message_size);
                    return 1;
                }
                if (cudaMalloc((void **) &ts_gpubuf, options.max_message_size)) {
                    fprintf(stderr, "Error allocating sending tGPU memory %lu\n",
                            options.max_message_size);
                    return 1;
                }
            }
        } else if (options.src == 'D' && options.dst == 'D') {
            if (myid == 0) {
                printf("** GPU to GPU\n");
            }
            is_dd = 1;
        }
    } else if (options.cpy_from_c) {
        if ((r_cpubuf =
            (char *) malloc(options.max_message_size)) == NULL) {
            fprintf(stderr, "Error allocating receiving CPU memory %lu\n",
                    options.max_message_size);
            return 1;
        }
        if ((s_cpubuf =
            (char *) malloc(options.max_message_size)) == NULL){
            fprintf(stderr, "Error allocating sending CPU memory %lu\n",
                    options.max_message_size);
            return 1;
        }

        if (options.src == 'H' && options.dst == 'H') {
            if (myid == 0) {
                printf("** Host to Host (Copy H to H)\n");
            }
            is_hh = 1;
        } else if (options.src == 'D' && options.dst == 'D') {
            if (myid == 0) {
                printf("** Device to Device (Copy H to D)\n");
            }
            is_dd = 1;
        }
    }

    print_header(myid, LAT);

    /* Latency test */
    for(size = options.min_message_size; size <= options.max_message_size; size = (size ? size * 2 : 1)) {
        if (options.cpy_from_d && (is_hh || is_dd)) {
            if (is_hh) {
              options.src = 'D';
              options.dst = 'D';
              set_buffer_pt2pt(s_gpubuf, myid, options.accel, 'a', size);
              options.src = 'H';
              options.dst = 'H';
            } else {
              set_buffer_pt2pt(s_gpubuf, myid, options.accel, 'a', size);
            }
        } else if (options.cpy_from_c && (is_hh || is_dd)) {
            if (is_hh) {
              set_buffer_pt2pt(s_cpubuf, myid, options.accel, 'a', size);
            } else {
              options.src = 'H';
              options.dst = 'H';
              set_buffer_pt2pt(s_cpubuf, myid, options.accel, 'a', size);
              options.src = 'D';
              options.dst = 'D';
            }
        } else {
            set_buffer_pt2pt(s_buf, myid, options.accel, 'a', size);
        }
        set_buffer_pt2pt(r_buf, myid, options.accel, 'b', size);

        if(size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        if(myid == 0) {
            for(i = 0; i < options.iterations + options.skip; i++) {
                if(i == options.skip) {
                    t_start = MPI_Wtime();
                }

                if (options.cpy_from_d) {
                    if (is_hh) {
                        if (options.add_serial) {
                            cudaMemcpy(ts_gpubuf, s_gpubuf,
                                       size, cudaMemcpyDeviceToDevice);
                            cudaMemcpy(s_buf, ts_gpubuf,
                                       size, cudaMemcpyDeviceToHost);
                        } else {
                            cudaMemcpy(s_buf, s_gpubuf,
                                       size, cudaMemcpyDeviceToHost);
                        }
                    } else if (is_dd) {
                        cudaMemcpy(s_buf, s_gpubuf, size, cudaMemcpyDeviceToDevice);
                    }
                } else if (options.cpy_from_c) {
                    if (is_hh) {
                        memcpy(s_buf, s_cpubuf, size);
                    } else if (is_dd) {
                        cudaMemcpy(s_buf, s_cpubuf, size, cudaMemcpyHostToDevice);
                    }
                }

                MPI_CHECK(MPI_Send(s_buf, size, MPI_CHAR, 1, 1, MPI_COMM_WORLD));
                MPI_CHECK(MPI_Recv(r_buf, size, MPI_CHAR, 1, 1, MPI_COMM_WORLD, &reqstat));

                if (options.cpy_from_d) {
                    if (is_hh) {
                        if (options.add_serial) {
                            cudaMemcpy(r_gpubuf, r_buf,
                                       size, cudaMemcpyHostToDevice);
                            cudaMemcpy(tr_gpubuf, r_gpubuf,
                                       size, cudaMemcpyDeviceToDevice);
                        } else {
                            cudaMemcpy(r_gpubuf, r_buf,
                                       size, cudaMemcpyHostToDevice);
                        }
                    } else if (is_dd) {
                        cudaMemcpy(r_gpubuf, r_buf, size, cudaMemcpyDeviceToDevice);
                    }
                } else if (options.cpy_from_c) {
                    if (is_hh) {
                        memcpy(r_cpubuf, r_buf, size);
                    } else if (is_dd) {
                        cudaMemcpy(r_cpubuf, r_buf, size, cudaMemcpyDeviceToHost);
                    }
                }
            }

            t_end = MPI_Wtime();
        }

        else if(myid == 1) {
            for(i = 0; i < options.iterations + options.skip; i++) {
                MPI_CHECK(MPI_Recv(r_buf, size, MPI_CHAR, 0, 1, MPI_COMM_WORLD, &reqstat));
                if (options.cpy_from_d) {
                    if (is_hh) {
                        if (options.add_serial) {
                            cudaMemcpy(r_gpubuf, r_buf,
                                       size, cudaMemcpyHostToDevice);
                            cudaMemcpy(tr_gpubuf, r_gpubuf,
                                       size, cudaMemcpyDeviceToDevice);
                        } else {
                            cudaMemcpy(r_gpubuf, r_buf,
                                       size, cudaMemcpyHostToDevice);
                        }
                    } else if (is_dd) {
                        cudaMemcpy(r_gpubuf, r_buf, size, cudaMemcpyDeviceToDevice);
                    }
                } else if (options.cpy_from_c) {
                    if (is_hh) {
                        memcpy(r_cpubuf, r_buf, size);
                    } else if (is_dd) {
                        cudaMemcpy(r_cpubuf, r_buf, size, cudaMemcpyDeviceToHost);
                    }
                }

                if (options.cpy_from_d) {
                    if (is_hh) {
                        if (options.add_serial) {
                            cudaMemcpy(ts_gpubuf, s_gpubuf,
                                       size, cudaMemcpyDeviceToDevice);
                            cudaMemcpy(s_buf, ts_gpubuf,
                                       size, cudaMemcpyDeviceToHost);
                        } else {
                            cudaMemcpy(s_buf, s_gpubuf,
                                       size, cudaMemcpyDeviceToHost);
                        }
                    } else if (is_dd) {
                        cudaMemcpy(s_buf, s_gpubuf, size, cudaMemcpyDeviceToDevice);
                    }
                } else if (options.cpy_from_c) {
                    if (is_hh) {
                        memcpy(s_buf, s_cpubuf, size);
                    } else if (is_dd) {
                        cudaMemcpy(s_buf, s_cpubuf, size, cudaMemcpyHostToDevice);
                    }
                }

                MPI_CHECK(MPI_Send(s_buf, size, MPI_CHAR, 0, 1, MPI_COMM_WORLD));
            }
        }

        if(myid == 0) {
            double latency = (t_end - t_start) * 1e6 / (2.0 * options.iterations);

            fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
                    FLOAT_PRECISION, latency);
            fflush(stdout);
        }
    }

    free_memory(s_buf, r_buf, myid);

    if (options.cpy_from_d) {
        cudaFree(r_gpubuf);
        cudaFree(s_gpubuf);
        if (options.add_serial) {
            cudaFree(tr_gpubuf);
            cudaFree(ts_gpubuf);
        }
    } else if (options.cpy_from_c) {
        free(r_cpubuf);
        free(s_cpubuf);
    }
    MPI_CHECK(MPI_Finalize());

    if (NONE != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }

    return EXIT_SUCCESS;
}

