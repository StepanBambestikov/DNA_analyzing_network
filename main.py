import pandas as pd

import classes3
import model_selection as ms
import network_classes as networks
import loss
import classes2
import Na_loader

# model_types = [networks.linear_classifier, networks.SimpleNetwork, networks.two_layer_network]
# model_names = ["linear_classifier", "SimpleNetwork", "two_layer_network"]
# conv_factors = [False, False, False]
# model_types = [networks.dna_2d_linear_classifier]
# model_names = ["dna_2d_linear_classifier"]
# conv_factors = [False]
# model_types = [networks.ConvNetwork2d, networks.dna_1d_linear_classifier]
# model_names = ["lConvNetwork2d", "dna_2d_linear_classifier"]
# conv_factors = [True, False]
model_name = "conv2d_net_18_05"
model_types = [classes3.conv2d_Mg_net_24_05]
model_names = [model_name]
conv_factors = [True]
origin_data_type = ms.Data_types.Na_D2
#
# model_types_evaluations = ms.models_analysis(model_types, model_names, conv_factors, origin_data_type,
#                                              ms.train_and_evaluate_models_for_all_parameters)
# ms.predictions_plotting(model_types_evaluations)
loss_name = "l1_loss_absolute_Tm"
losses = {loss_name: loss.normalized_l1_multi_loss_30_04_relative_Tm}

model_types_evaluations, dna_activity_columns, models = ms.models_analysis(model_types, model_names, conv_factors,
                                                                           origin_data_type,
                                                                           ms.train_and_evaluate_multi_model, losses,
                                                                           train_val_splitter=Na_loader.multi_splitter)
ms.predictions_plotting(model_types_evaluations)
ms.save_model(models[0], model_name + loss_name)
ms.save_predictions(model_types_evaluations, "_save.xlsx", pandas_load_method=pd.DataFrame.to_excel,
                    train_additive_columns=dna_activity_columns[0],
                    val_additive_columns=dna_activity_columns[1])
