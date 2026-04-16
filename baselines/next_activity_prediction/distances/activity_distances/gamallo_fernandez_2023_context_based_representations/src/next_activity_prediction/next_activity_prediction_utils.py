from distances.activity_distances.bose_2009_context_aware_trace_clustering.algorithm import \
    get_substitution_and_insertion_scores
from evaluation.data_util.util_activity_distances import get_alphabet





def transform_bose_similarity_to_distance(bose_similarity_matrix):
    #high bose similarity scores are small distances
    bose_similarity_matrix = {key: -value for key, value in bose_similarity_matrix.items()}

    #transform similarity scores to distances
    min_value = -1*min(bose_similarity_matrix.values())

    distance_matrix = {key: value + min_value for key, value in
                       bose_similarity_matrix.items()}

    return distance_matrix

b =  {('Analyze Defect', 'Analyze Defect'): 0.292081962513166, ('Analyze Defect', 'Archive Repair'): 0.7113886673756609, ('Analyze Defect', 'Inform User'): 0.4795929812226674, ('Analyze Defect', 'Register'): 0.6794341134163684, ('Analyze Defect', 'Repair (Complex):0'): 0.4547316277165516, ('Analyze Defect', 'Repair (Complex):1'): 0.4506556159670717, ('Analyze Defect', 'Repair (Simple)'): 0.477012223257239, ('Analyze Defect', 'Restart Repair'): 0.905852237260829, ('Analyze Defect', 'Test Repair'): 0.6083207000563228, ('Archive Repair', 'Analyze Defect'): 0.7113886673756609, ('Archive Repair', 'Archive Repair'): 0.2020590131205277, ('Archive Repair', 'Inform User'): 0.470901838626663, ('Archive Repair', 'Register'): 0.7589557002626287, ('Archive Repair', 'Repair (Complex):0'): 0.828673925685188, ('Archive Repair', 'Repair (Complex):1'): 0.8261549247968772, ('Archive Repair', 'Repair (Simple)'): 0.7201030967194502, ('Archive Repair', 'Restart Repair'): 0.9853738241070894, ('Archive Repair', 'Test Repair'): 0.5113251466971784, ('Inform User', 'Analyze Defect'): 0.4795929812226674, ('Inform User', 'Archive Repair'): 0.470901838626663, ('Inform User', 'Inform User'): 0.2919974862483837, ('Inform User', 'Register'): 0.7735818761555394, ('Inform User', 'Repair (Complex):0'): 0.5393936172479541, ('Inform User', 'Repair (Complex):1'): 0.5439741829754934, ('Inform User', 'Repair (Simple)'): 0.5431512621841318, ('Inform User', 'Restart Repair'): 1.0, ('Inform User', 'Test Repair'): 0.40786956414011466, ('Register', 'Analyze Defect'): 0.6794341134163684, ('Register', 'Archive Repair'): 0.7589557002626287, ('Register', 'Inform User'): 0.7735818761555394, ('Register', 'Register'): 0.21718666771209524, ('Register', 'Repair (Complex):0'): 0.7967193717258956, ('Register', 'Repair (Complex):1'): 0.7942003708375847, ('Register', 'Repair (Simple)'): 0.6881485427601577, ('Register', 'Restart Repair'): 0.9534192701477969, ('Register', 'Test Repair'): 0.6558877329432906, ('Repair (Complex):0', 'Analyze Defect'): 0.4547316277165516, ('Repair (Complex):0', 'Archive Repair'): 0.828673925685188, ('Repair (Complex):0', 'Inform User'): 0.5393936172479541, ('Repair (Complex):0', 'Register'): 0.7967193717258956, ('Repair (Complex):0', 'Repair (Complex):0'): 0.282010577341566, ('Repair (Complex):0', 'Repair (Complex):1'): 0.3523291967580394, ('Repair (Complex):0', 'Repair (Simple)'): 0.4644493027429295, ('Repair (Complex):0', 'Restart Repair'): 0.619646926790322, ('Repair (Complex):0', 'Test Repair'): 0.45839884257488306, ('Repair (Complex):1', 'Analyze Defect'): 0.4506556159670717, ('Repair (Complex):1', 'Archive Repair'): 0.8261549247968772, ('Repair (Complex):1', 'Inform User'): 0.5439741829754934, ('Repair (Complex):1', 'Register'): 0.7942003708375847, ('Repair (Complex):1', 'Repair (Complex):0'): 0.3523291967580394, ('Repair (Complex):1', 'Repair (Complex):1'): 0.2829521398565467, ('Repair (Complex):1', 'Repair (Simple)'): 0.46888451180766516, ('Repair (Complex):1', 'Restart Repair'): 0.5644818551256159, ('Repair (Complex):1', 'Test Repair'): 0.4584199591881272, ('Repair (Simple)', 'Analyze Defect'): 0.477012223257239, ('Repair (Simple)', 'Archive Repair'): 0.7201030967194502, ('Repair (Simple)', 'Inform User'): 0.5431512621841318, ('Repair (Simple)', 'Register'): 0.6881485427601577, ('Repair (Simple)', 'Repair (Complex):0'): 0.4644493027429295, ('Repair (Simple)', 'Repair (Complex):1'): 0.46888451180766516, ('Repair (Simple)', 'Repair (Simple)'): 0.3543122814218801, ('Repair (Simple)', 'Restart Repair'): 0.5218214740703392, ('Repair (Simple)', 'Test Repair'): 0.41345592628724975, ('Restart Repair', 'Analyze Defect'): 0.905852237260829, ('Restart Repair', 'Archive Repair'): 0.9853738241070894, ('Restart Repair', 'Inform User'): 1.0, ('Restart Repair', 'Register'): 0.9534192701477969, ('Restart Repair', 'Repair (Complex):0'): 0.619646926790322, ('Restart Repair', 'Repair (Complex):1'): 0.5644818551256159, ('Restart Repair', 'Repair (Simple)'): 0.5218214740703392, ('Restart Repair', 'Restart Repair'): 0.0, ('Restart Repair', 'Test Repair'): 0.7463049982639784, ('Test Repair', 'Analyze Defect'): 0.6083207000563228, ('Test Repair', 'Archive Repair'): 0.5113251466971784, ('Test Repair', 'Inform User'): 0.40786956414011466, ('Test Repair', 'Register'): 0.6558877329432906, ('Test Repair', 'Repair (Complex):0'): 0.45839884257488306, ('Test Repair', 'Repair (Complex):1'): 0.4584199591881272, ('Test Repair', 'Repair (Simple)'): 0.41345592628724975, ('Test Repair', 'Restart Repair'): 0.7463049982639784, ('Test Repair', 'Test Repair'): 0.3518321788400499}

transform_bose_similarity_to_distance(b)

"""
def get_embeddings(activity_distance_functions, log,
                                      n_gram_size_bose_2009=3):
    for activity_distance_function in activity_distance_functions:
        if "Bose 2009 Substitution Scores" == activity_distance_function:
            bose_similarity_matrix = get_substitution_and_insertion_scores(
                log,
                get_alphabet(
                    log), n_gram_size_bose_2009)
            distance_matrix = transform_bose_similarity_to_distance(bose_similarity_matrix)
            bose_embeddings = transform_distances_to_embeddings()

        elif "De Koninck 2018 act2vec" == activity_distance_function[:23]:
            if activity_distance_function[24:] == "CBOW":
                sg = 0
            else:
                sg = 1
                embeddings = get_act2vec_distance_matrix(logs_with_replaced_activities_dict[key],
                                                                      get_alphabet(
                                                                          logs_with_replaced_activities_dict[key]), sg)
                activity_distance_matrix_dict[activity_distance_function][key] = act2vec_distance_matrix
        elif "Unit Distance" == activity_distance_function:
            for key in logs_with_replaced_activities_dict:
                unit_distance_matrix = get_unit_cost_activity_distance_matrix(logs_with_replaced_activities_dict[key], get_alphabet(
                                                                          logs_with_replaced_activities_dict[key]))
                activity_distance_matrix_dict[activity_distance_function][key] = unit_distance_matrix
        elif "Chiorrini 2022 Embedding Process Structure" == activity_distance_function:
            for key in logs_with_replaced_activities_dict:
                embedding_process_structure_distance_matrix = get_embedding_process_structure_distance_matrix(logs_with_replaced_activities_dict[key],
                                                                      get_alphabet(
                                                                          logs_with_replaced_activities_dict[key]), False)
                activity_distance_matrix_dict[activity_distance_function][key] = embedding_process_structure_distance_matrix
        elif "Gamallo Fernandez 2023 Context Based" == activity_distance_function:
            for key in logs_with_replaced_activities_dict:
                embedding_process_structure_distance_matrix = get_context_based_distance_matrix(
                    logs_with_replaced_activities_dict[key])
                activity_distance_matrix_dict[activity_distance_function][
                    key] = embedding_process_structure_distance_matrix
    return dict(activity_distance_matrix_dict)
"""