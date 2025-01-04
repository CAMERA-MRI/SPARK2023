if __name__ == '__main__':
    from nnunetv2.paths import nnUNet_results, nnUNet_raw
    import torch
    from batchgenerators.utilities.file_and_folder_operations import join
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    # nnUNetv2_predict -d 3 -f 0 -c 3d_lowres -i imagesTs -o imagesTs_predlowres --continue_prediction

    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset500_BraTS2023/nnUNetTrainer_2epochs__nnUNetPlans__3d_fullres'),
        use_folds=(0,), 
        checkpoint_name='checkpoint_final.pth',
    )
    # variant 1: give input and output folders
    predictor.predict_from_files(join(nnUNet_raw, 'Dataset500_BraTS2023/imagesTs'),
                                 join(nnUNet_raw, 'Dataset500_BraTS2023/imagesTs_predfullres'),
                                 save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

    # variant 2, use list of files as inputs. Note how we use nested lists!!!
    indir = join(nnUNet_raw, 'Dataset500_BraTS2023/imagesTs')
    outdir = join(nnUNet_raw, 'Dataset500_BraTS2023/imagesTs_predfullres')    # 4 '0000
    predictor.predict_from_files([[join(indir, 'BraTS-SSA-00192-000-0000.nii.gz')],
                                  [join(indir, 'BraTS-SSA-00198-000-0000.nii.gz')]],
                                 [join(outdir, 'BraTS-SSA-00192-000-0000.nii.gz'),
                                  join(outdir, 'BraTS-SSA-00198-000-0000.nii.gz')],
                                 save_probabilities=False, overwrite=True,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

    # variant 2.5, returns segmentations
    indir = join(nnUNet_raw, 'Dataset500_BraTS2023/imagesTs')
    predicted_segmentations = predictor.predict_from_files([[join(indir, 'BraTS-SSA-00198-000-0000.nii.gz')],
                                                            [join(indir, 'BraTS-SSA-00210-000-0000.nii.gz')]],
                                                           None,
                                                           save_probabilities=True, overwrite=True,
                                                           num_processes_preprocessing=2,
                                                           num_processes_segmentation_export=2,
                                                           folder_with_segs_from_prev_stage=None, num_parts=1,
                                                           part_id=0)

    # predict several npy images
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    img, props = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset500_BraTS2023/imagesTs/BraTS-GLI-01710-000-0000.nii.gz')])
    img2, props2 = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset500_BraTS2023/imagesTs/BraTS-GLI-01711-000-0000.nii.gz')])
    img3, props3 = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset500_BraTS2023/imagesTs/BraTS-GLI-01712-000-0000.nii.gz')])
    img4, props4 = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset500_BraTS2023/imagesTs/BraTS-GLI-01713-000-0000.nii.gz')])
    # we do not set output files so that the segmentations will be returned. You can of course also specify output
    # files instead (no return value on that case)
    ret = predictor.predict_from_list_of_npy_arrays([img, img2, img3, img4],
                                                    None,
                                                    [props, props2, props3, props4],
                                                    None, 2, save_probabilities=False,
                                                    num_processes_segmentation_export=2)

    # predict a single numpy array
    img, props = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset500_BraTS2023/imagesTs/BraTS-SSA-00188-000-0000.nii.gz')])
    ret = predictor.predict_single_npy_array(img, props, None, None, True)

    # custom iterator

    img, props = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset500_BraTS2023/imagesTs/BraTS-GLI-00573-000-0000.nii.gz')])
    img2, props2 = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset500_BraTS2023/imagesTs/BraTS-GLI-00585-000-0000.nii.gz')])
    img3, props3 = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset500_BraTS2023/imagesTs/BraTS-GLI-00592-000-0000.nii.gz')])
    img4, props4 = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset500_BraTS2023/imagesTs/BraTS-GLI-00595-000-0000.nii.gz')])


    # each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properites' keys!
    # If 'ofile' is None, the result will be returned instead of written to a file
    # the iterator is responsible for performing the correct preprocessing!
    # note how the iterator here does not use multiprocessing -> preprocessing will be done in the main thread!
    # take a look at the default iterators for predict_from_files and predict_from_list_of_npy_arrays
    # (they both use predictor.predict_from_data_iterator) for inspiration!
    def my_iterator(list_of_input_arrs, list_of_input_props):
        preprocessor = predictor.configuration_manager.preprocessor_class(verbose=predictor.verbose)
        for a, p in zip(list_of_input_arrs, list_of_input_props):
            data, seg = preprocessor.run_case_npy(a,
                                                  None,
                                                  p,
                                                  predictor.plans_manager,
                                                  predictor.configuration_manager,
                                                  predictor.dataset_json)
            yield {'data': torch.from_numpy(data).contiguous().pin_memory(), 'data_properties': p, 'ofile': None}


    ret = predictor.predict_from_data_iterator(my_iterator([img, img2, img3, img4], [props, props2, props3, props4]),
                                               save_probabilities=False, num_processes_segmentation_export=3)
