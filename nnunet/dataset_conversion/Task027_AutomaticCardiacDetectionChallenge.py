#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from collections import OrderedDict
from nnunet.common import *
import shutil
import numpy as np
from sklearn.model_selection import KFold
import SimpleITK as sitk


def convert_to_submission(source_dir, target_dir):
    niftis = subfiles(source_dir, join=False, suffix=".nii.gz")
    patientids = np.unique([i[:10] for i in niftis])
    maybe_mkdir_p(target_dir)
    for p in patientids:
        files_of_that_patient = subfiles(source_dir, prefix=p, suffix=".nii.gz", join=False)
        assert len(files_of_that_patient)
        files_of_that_patient.sort()
        # first is ED, second is ES
        shutil.copy(join(source_dir, files_of_that_patient[0]), join(target_dir, p + "_ED.nii.gz"))
        shutil.copy(join(source_dir, files_of_that_patient[1]), join(target_dir, p + "_ES.nii.gz"))


def get_4d_nii():
    folder = r"\\SEUVCL-DATA-01\Medical\ACDC\training\training"
    folder_test = r"\\SEUVCL-DATA-01\Medical\ACDC\training\training"
    out_folder = r"D:\XiaohanYuan\nnUnet_OnWindows\nnunet\Data\nnUNet_raw_data_base\nnUNet_raw_data\Task027_ACDC"

    maybe_mkdir_p(join(out_folder, "imagesTr"))
    maybe_mkdir_p(join(out_folder, "imagesTs"))
    maybe_mkdir_p(join(out_folder, "labelsTr"))

    # train
    all_train_files = []
    patient_dirs_train = subfolders(folder, prefix="patient")
    for p in patient_dirs_train:
        current_dir = p
        data_files_train = [i for i in subfiles(current_dir, suffix=".nii.gz") if i.find("_gt") == -1 and i.find("_4d") == -1]
        corresponding_seg_files = [i[:-7] + "_gt.nii.gz" for i in data_files_train]
        for d, s in zip(data_files_train, corresponding_seg_files):
            patient_identifier = d.split("\\")[-1][:-7]
            all_train_files.append(patient_identifier + "_0000.nii.gz")
            # shutil.copy(d, join(out_folder, "imagesTr", patient_identifier + "_0000.nii.gz"))
            # shutil.copy(s, join(out_folder, "labelsTr", patient_identifier + ".nii.gz"))

    # test
    all_test_files = []
    patient_dirs = os.listdir(folder_test)
    for patient_identifier in patient_dirs:
        input_filename = os.path.join(folder_test, patient_identifier, patient_identifier + "_4d.nii.gz")
        if os.path.exists(os.path.join(folder_test, patient_identifier, patient_identifier + "_frame01" + ".nii.gz")):
            input_frame01 = os.path.join(folder_test, patient_identifier, patient_identifier + "_frame01" + ".nii.gz")
        else:
            input_frame01 = os.path.join(folder_test, patient_identifier, patient_identifier + "_frame04" + ".nii.gz")

        print(patient_identifier)
        f_path = os.path.join(input_filename)
        img_in = sitk.ReadImage(f_path)
        img_npy = sitk.GetArrayFromImage(img_in)

        img01 = sitk.ReadImage(input_frame01)
        print(img_npy.shape)

        for i in range(img_npy.shape[0]):
            output_filename = os.path.join(out_folder, "imagesTs", patient_identifier + "_frame%02d" % (i+1) + "_0000.nii.gz")
            bg_data = sitk.GetImageFromArray(img_npy[i, :, :, :])
            bg_data.SetOrigin(img01.GetOrigin())
            bg_data.SetDirection(img01.GetDirection())
            bg_data.SetSpacing(img01.GetSpacing())
            sitk.WriteImage(bg_data, output_filename)
            all_test_files.append(patient_identifier + "_frame%02d" % (i+1) + "_0000.nii.gz")

    json_dict = OrderedDict()
    json_dict['name'] = "ACDC"
    json_dict['description'] = "cardias cine MRI segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see ACDC challenge"
    json_dict['licence'] = "see ACDC challenge"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "MRI",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "RV",
        "2": "MLV",
        "3": "LVC"
    }
    json_dict['numTraining'] = len(all_train_files)
    json_dict['numTest'] = len(all_test_files)
    json_dict['training'] = [{'image': ".\imagesTr\%s.nii.gz" % i.split("\\")[-1][:-12], "label": ".\labelsTr\%s.nii.gz" % i.split("\\")[-1][:-12]} for i in
                             all_train_files]
    json_dict['test'] = [".\imagesTs\%s.nii.gz" % i.split("\\")[-1][:-12] for i in all_test_files]

    save_json(json_dict, os.path.join(out_folder, "dataset.json"))

    # create a dummy split (patients need to be separated)
    splits = []
    patients = np.unique([i[:10] for i in all_train_files])
    patientids = [i[:-12] for i in all_train_files]

    kf = KFold(5, shuffle=True, random_state=12345)
    for tr, val in kf.split(patients):
        splits.append(OrderedDict())
        tr_patients = patients[tr]
        splits[-1]['train'] = [i[:-12] for i in all_train_files if i[:10] in tr_patients]
        val_patients = patients[val]
        splits[-1]['val'] = [i[:-12] for i in all_train_files if i[:10] in val_patients]

    # you should move this file to nnUNet_preprocessed fold for training
    save_pickle(splits, r".\Data\nnUNet_raw_data_base\nnUNet_preprocessed\Task027_ACDC\splits_final.pkl")




if __name__ == "__main__":
    get_4d_nii()

    # folder = r"\\SEUVCL-DATA-01\Medical\ACDC\training\training"
    # folder_test = r"\\SEUVCL-DATA-01\Medical\ACDC\testing\testing"
    # out_folder = r"D:\XiaohanYuan\nnUnet_OnWindows\nnunet\Data\nnUNet_raw_data_base\nnUNet_raw_data\Task027_ACDC"
    #
    # maybe_mkdir_p(join(out_folder, "imagesTr"))
    # maybe_mkdir_p(join(out_folder, "imagesTs"))
    # maybe_mkdir_p(join(out_folder, "labelsTr"))
    #
    # # train
    # all_train_files = []
    # patient_dirs_train = subfolders(folder, prefix="patient")
    # for p in patient_dirs_train:
    #     current_dir = p
    #     data_files_train = [i for i in subfiles(current_dir, suffix=".nii.gz") if i.find("_gt") == -1 and i.find("_4d") == -1]
    #     corresponding_seg_files = [i[:-7] + "_gt.nii.gz" for i in data_files_train]
    #     for d, s in zip(data_files_train, corresponding_seg_files):
    #         patient_identifier = d.split("\\")[-1][:-7]
    #         all_train_files.append(patient_identifier + "_0000.nii.gz")
    #         shutil.copy(d, join(out_folder, "imagesTr", patient_identifier + "_0000.nii.gz"))
    #         shutil.copy(s, join(out_folder, "labelsTr", patient_identifier + ".nii.gz"))
    #
    # # test
    # all_test_files = []
    # patient_dirs_test = subfolders(folder_test, prefix="patient")
    # for p in patient_dirs_test:
    #     current_dir = p
    #     data_files_test = [i for i in subfiles(current_dir, suffix=".nii.gz") if i.find("_gt") == -1 and i.find("_4d") == -1]
    #     for d in data_files_test:
    #         patient_identifier = d.split("\\")[-1][:-7]
    #         all_test_files.append(patient_identifier + "_0000.nii.gz")
    #         shutil.copy(d, join(out_folder, "imagesTs", patient_identifier + "_0000.nii.gz"))
    #
    #
    # json_dict = OrderedDict()
    # json_dict['name'] = "ACDC"
    # json_dict['description'] = "cardias cine MRI segmentation"
    # json_dict['tensorImageSize'] = "4D"
    # json_dict['reference'] = "see ACDC challenge"
    # json_dict['licence'] = "see ACDC challenge"
    # json_dict['release'] = "0.0"
    # json_dict['modality'] = {
    #     "0": "MRI",
    # }
    # json_dict['labels'] = {
    #     "0": "background",
    #     "1": "RV",
    #     "2": "MLV",
    #     "3": "LVC"
    # }
    # json_dict['numTraining'] = len(all_train_files)
    # json_dict['numTest'] = len(all_test_files)
    # json_dict['training'] = [{'image': ".\imagesTr\%s.nii.gz" % i.split("\\")[-1][:-12], "label": ".\labelsTr\%s.nii.gz" % i.split("\\")[-1][:-12]} for i in
    #                          all_train_files]
    # json_dict['test'] = [".\imagesTs\%s.nii.gz" % i.split("\\")[-1][:-12] for i in all_test_files]
    #
    # save_json(json_dict, os.path.join(out_folder, "dataset.json"))
    #
    # # create a dummy split (patients need to be separated)
    # splits = []
    # patients = np.unique([i[:10] for i in all_train_files])
    # patientids = [i[:-12] for i in all_train_files]
    #
    # kf = KFold(5, shuffle=True, random_state=12345)
    # for tr, val in kf.split(patients):
    #     splits.append(OrderedDict())
    #     tr_patients = patients[tr]
    #     splits[-1]['train'] = [i[:-12] for i in all_train_files if i[:10] in tr_patients]
    #     val_patients = patients[val]
    #     splits[-1]['val'] = [i[:-12] for i in all_train_files if i[:10] in val_patients]
    #
    # # you should move this file to nnUNet_preprocessed fold for training
    # save_pickle(splits, r".\Data\nnUNet_raw_data_base\nnUNet_preprocessed\Task027_ACDC\splits_final.pkl")