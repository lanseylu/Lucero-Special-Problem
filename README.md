> Modelling COVID-19 in Palengkes: An Agent-based Case Study of Baguio City's Public Market
> Author: Lanchelot Domagsang Lucero

How to use this model:
1. Getting started
    - Open cmd, write and execute the ff code: pip install covid19-supermarket-abm
    - This package requires Python >= 3.6

2. Running experiments
    The model has four main inputs, a config file, a graph G, a path generator, and the arguments for the path generator.
    - Config file:
        - arrival_rate - rate at which buyers arrive at the markets
        - traversal_time - average time of a buyer per node in the markets
        - num_hours_open - market's operation num_hours_open
        - infection_proportion - the proportion of buyers infected in the market
        - max_customers_in_store - max. number of buyers allowed in the market (optional)
        - with_node_capacity - true if a node can only have a defined max. of buyers allowed per node (optional)
        - node_capacity - defined number of max. no. of buyers allowed per node in the market

    b. Graph G
        - We use networkx package to create the market network. First, we need to specify the (x,y) coordinates of each node. So in a very simple example, we have four nodes, arranged in a square at with coordinates (0,0), (0,1), (1,0), and (1,1). we code this as: pos = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
        - Next, we need to specify the edges in the network; in other words, which nodes are connected to each other.We code this as: edges = [(0,1), (1,3), (0,2), (2,3)]
        - We create the graph as follows: 
            - from covid19_supermarket_abm.utils.create_store_network import create_store_network
            - G = create_store_network(pos, edges)
        - To visualize your network, you can use nx.draw_networkx:
            - import networkx as nx
            - nx.draw_networkx(G, pos=pos, node_color='y')
        - To create a one-way setup, simply use directed=True: G = create_store_network(pos, edges, directed=True) 

    c. Path generator and Args
        - In this model, we used the synthetic_path_gen.py to create 1,000,000 full shopping trips. The shopping trips is named as 10^6.json (or 10^6_DIRECTIONAL.json if you uncommented it within the file).
        - Then in the simulation proper, in main.py, we used the function:
        - path_generator_function, path_generator_args = get_path_generator(path_generation='empirical', full_paths=full_paths) 
        - to generate the path generator (path_generator_function) and the arguments for it (path_generator_args)
    
4. Working with results
    - The model results are stored in results which is a dictionary tuple with 3 elements.
        - First element is a dataframe with most simulation metrics:
            - num_cust 	Total number of customers
            - num_S	Number of susceptible customers
            - num_I	Number of infected customers
            - total_exposure_time	Cumulative exposure time
            - num_contacts_per_cust	List of number of contacts with infectious customers per susceptible customer with at least one contact
            - num_cust_w_contact	Number of susceptible customers which have at least one contact with an infectious customer
            - mean_num_cust_in_store	Mean number of customers in the store during the simulation
            - max_num_cust_in_store	Maximum number of customers in the store during the simulation
            - num_contacts	Total number of contacts between infectious customers and susceptible customers
            - mean_shopping_time	Mean of the shopping times
            - num_waiting_people	Number of people who are queueing outside at every minute of the simulation (when the number of customers in the store is restricted)
            - mean_waiting_time	Mean time that customers wait before being allowed to enter (when the number of customers in the store is restricted)
            - store_open_length	Length of the store's opening hours (in minutes)
            - total_time_crowded	Total time that nodes were crowded (when there are more than thres number of customers in a node. Default value of thres is 4)
            - exposure_times	List of exposure times of customers (only recording positive exposure times)
        - The second element gives the number of encounters or contacts per node
        - Third element is a dataframe containing the exposure time per node
    b. In the model, we list all used shopping trips by buyers across the 1000 simulations to all_buyer_paths.txt to help with processing the average number of unique buyers per node in our study (which is around 2 million paths). Between simulations, a string called "Market closed." is indicated. 

References:
F. Ying and N. O’Clery, Modelling covid-19 transmission in super-
markets using an agent-based model, PLOS ONE, 16 (2021), p. e0249821.
doi:http://10.1371/journal.pone.0249821.
