def heat_map(start, stop, x, shap_values, var_name='Feature 1', plot_type='bar', title=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.colors import BoundaryNorm
    from textwrap import wrap
    import numpy as np; np.random.seed(1)
    
    if plot_type=='heat':
        ## ColorMap-------------------------
        # define the colormap
        cmap = plt.get_cmap('PuOr_r')

        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # create the new map
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

        # define the bins and normalize and forcing 0 to be part of the colorbar!
        bounds = np.arange(np.min(shap_values),np.max(shap_values),.005)
        idx=np.searchsorted(bounds,0)
        bounds=np.insert(bounds,idx,0)
        norm = BoundaryNorm(bounds, cmap.N)
        ##------------------------------------
    
    if title is None: title = '\n'.join(wrap('{} values and contribution scores'.format(var_name), width=40))
    
    if plot_type=='heat' or plot_type=='heat_abs':
        plt.rcParams["figure.figsize"] = 9,3
        if plot_type=='heat_abs':
            shap_values = np.absolute(shap_values)
            cmap = 'Reds'
        fig, ax1 = plt.subplots(sharex=True)
        extent = [start, stop, -2, 2]
        im1 = ax1.imshow(shap_values[np.newaxis, :], cmap=cmap, norm=norm, aspect="auto", extent=extent)
        ax1.set_yticks([])
        ax1.set_xlim(extent[0], extent[1])
        ax1.title.set_text(title)
        fig.colorbar(im1, ax=ax1, pad=0.1)
        ax2 = ax1.twinx()
        ax2.plot(np.arange(start, stop), x, color='black')
    elif plot_type=='bar':
        plt.rcParams["figure.figsize"] = 8,3
        fig, ax1 = plt.subplots(sharex=True)
        mask1 = shap_values < 0
        mask2 = shap_values >= 0
        ax1.bar(np.arange(start, stop)[mask1], shap_values[mask1], color='blue', label='Negative Shapely values')
        ax1.bar(np.arange(start, stop)[mask2], shap_values[mask2], color='red', label='Positive Shapely values')
        ax1.set_title(title)
        ax2 = ax1.twinx()
        ax2.plot(np.arange(start, stop), x, 'k-', label='Sequential data points')
        # legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
    
    ax1.set_xlabel('Time steps')
    if plot_type=='bar': ax1.set_ylabel('Shapely values')
    ax2.set_ylabel(var_name + ' sequential values')
    plt.tight_layout()
    plt.show()


def heat_map_all_features(start, stop, shap_values, num_feature=20, var_name='Feature 1', plot_type='bar', title=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.colors import BoundaryNorm
    from textwrap import wrap
    import numpy as np; np.random.seed(1)
    ## ColorMap-------------------------
    # define the colormap
    cmap = plt.get_cmap('PuOr_r')

    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize and forcing 0 to be part of the colorbar!
    bounds = np.arange(np.min(shap_values),np.max(shap_values),.005)
    idx=np.searchsorted(bounds,0)
    bounds=np.insert(bounds,idx,0)
    norm = BoundaryNorm(bounds, cmap.N)
    ##------------------------------------
    # Finding the top important features
    shap_values = shap_values.T
    inds = np.argsort(np.absolute(shap_values.sum(axis=1)))
    inds = inds[-num_feature:]
    inds = np.flip(inds)
    shap_values = shap_values[inds, :]
    var_name = np.array(var_name)[inds]
    
    if plot_type=='heat' or plot_type=='heat_abs':
        plt.rcParams["figure.figsize"] = 9,3
        if plot_type=='heat_abs':
            shap_values = np.absolute(shap_values)
            cmap = 'Reds'
            
        fig, ax = plt.subplots(1,1)
        extent = [start, stop, -1, 1]
        img = ax.imshow(shap_values, cmap=cmap, aspect='auto', extent=extent, interpolation='nearest')

        y_label_list = var_name
        x_label_list = np.arange(start, stop)

        ax.set_xlim(extent[0], extent[1])
        ax.set_yticks(np.linspace(start=1, stop=-1, num=len(y_label_list)))

        ax.set_yticklabels(y_label_list)

        ax.set_xlabel('Time steps')
        ax.set_title('Importance of all time steps for important features')

        fig.colorbar(img, ax=ax, pad=0.1)
        plt.show()