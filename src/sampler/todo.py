# TODO now:
# Check if pipeline (simulator), nodes (run_simu has simulator arg) are well implemented in parego and sao_optim and do final tests
# When finish clean wrapper_for_0d
# Check how nan in features is managed and in targets
# modify run_sim in parego
# Use fake_simulator in wrapper rather than saimultor_fake.py 

# TODO later:
# test sao_opt and parego with new map_dir (don't forget to enable use_simulator)
# delete sao_history_path from all sao_optim.json files

# TODO to_implement:
# ! compute metric of spreadness "distribution of distances" or "potential energy" or "distribution of segments/meshes/symplex"
# The transformation for f: R² -> R, so p=len(features)=2 and k=len(targets)=1
# 0. X_train.shape == (n, 2)
# 1. triangles = set_triangles([(x, y) for (x, y) in X_train])
#    # triangles.shape == (n, 3, 2) because for n points, there are n triangle meshes,
#    # each triangle has 3 points, each point has 2 dimensions (x, y)
# 2. triangles_up = [[(x, y, z=f(x, y)) for (x, y) in triangle] for triangle in triangles]
#    # triangles_up.shape == (n, 3, 3) because each triangle now has 3 points in 3D space (x, y, z)
# 3. areas = get_area_up(triangles_up)
#    # areas contains the surface area of each triangle in 3D space
# 4. score = mean(areas), std(areas)
#    # score is a tuple containing the mean and standard deviation of the surface areas

# ! if use_simulator=False, initialize data with fake_simu(lhs_data) instead of real_data

# TODO future work:
# set / find code to generate new lhs samples for test scores using sampler project
# How to sample more the interest zone for the test ?
# set test scores with focus on interest zone
# Add surface compute for targets space (and possibly for main features plane)
# ! "potential energy": asssimilates samples to atoms in R^(feature+targets) and computes potential energy of their interation that should be minimized
# discuss grid shape of EIC samples


"""
Names for my work
Based on your description, the technique you are using can be considered a variant of
Bayesian optimization, specifically focusing on maximizing the number of samples in an
interesting region of the design space. 
These names aim to capture the essence of your technique, which involves optimizing a
figure of merit (FOM) to find interesting samples in a design space using a combination
of GP standard deviation, probability of being in an interesting region, and coverage.

Here are five name suggestions for your technique:
1. Interest-Driven Global Optimization (IDGO):
Emphasizes that the optimization process is driven by the interest regions in the design space.
2. Targeted Bayesian Optimization (TBO):
Highlights the use of Bayesian optimization techniques with a focus on targeting specific regions of interest.
3. Focused Acquisition Sampling (FAS):
Indicates that the sampling process is focused on acquiring samples in areas of interest.
4. Region-Specific Global Optimization (RSGO):
Suggests that the optimization is specific to certain regions within the global design space.
5. Interest Region Sampling Optimization (IRSO):
Combines the concepts of sampling and optimization within regions of interest.

Here are five more name suggestions based on your description:
1. Interest-Driven Bayesian Optimization (IDBO)
2. Targeted Acquisition Bayesian Sampling (TABS)
3. Focused Bayesian Sampling (FBS)
4. Region-Specific Bayesian Optimization (RSBO)
5. Interest Area Bayesian Sampling (IABS)

Chosed name :
"Interest Region Bayesian Sampling" (IRBS) is a suitable name for your technique. It
effectively captures the essence of your approach, which involves using Bayesian
optimization to sample points in regions of interest within the design space. Here are
some reasons why this name works well:
Interest Region: Clearly indicates that the technique focuses on specific regions of
interest within the design space.
Bayesian: Highlights the use of Bayesian optimization principles, such as the Gaussian
Process (GP) and acquisition functions.
Sampling: Emphasizes the process of selecting or sampling points within the design space.

"""