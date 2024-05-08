
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class XAI_ALLCASES_PLOT():
    def __init__(self, data_dict, exp_name: str, threshold: float, embd_type = str):
        self.data_dict = data_dict
        self.exp_name = exp_name
        self.threshold = threshold
        self.embd_type = embd_type


        root = '/data1/fog/Hamid/SparkMET/EXPs/'
        name = 'train_prob_' + self.exp_name + '.csv'
        # self.df = pd.read_csv(os.path.join(root, self.exp_name, name))
        self.df = pd.read_csv('/data1/fog/Hamid/SparkMET/EXPs/05_PIT_V2_LRP_lr00001/train_prob_05_PIT_V2_LRP_lr00001.csv')

    def local_plot(self, date: str):

        hit, miss, fa, cr = self.get_predict_cases()

        lpr_matrix = self.data_dict[date]['lpr']#.data.cpu().numpy()

        fig, axs = plt.subplots(1, 1, figsize = (5, 4))
        axs.imshow(lpr_matrix)

        if date in hit:
            axs.set_title(f'{date}: Hit')
        elif date in miss:
            axs.set_title(f'{date}: Miss')
        elif date in fa:
            axs.set_title(f'{date}: False Alarm')
        else:
            axs.set_title(f'{date}: Correct Rejection')

    def _reshape_into_org(self, mtx):
        
        num_mtx = mtx.shape[1]
        output = []
        for n in range(num_mtx):
            expanded_matrix = np.zeros((32, 32))
            this_map = mtx[0, n, ...].flatten()
            # Fill each value into an 8x8 block
            for i, value in enumerate(this_map):
                # Calculate the starting row and column for the current block
                start_row = (i // 4) * 8
                start_col = (i % 4) * 8
                
                # Fill the 8x8 block with the current value
                expanded_matrix[start_row:start_row+8, start_col:start_col+8] = value

            output.append(expanded_matrix)

        return output

    def global_plot(self, scale: int):

        hit, miss, fa, cr = self.get_predict_cases()

        hit_cases_avg = self.process_category(hit, scale)
        miss_cases_avg = self.process_category(miss, scale)
        fa_cases_avg = self.process_category(fa, scale)
        cr_cases_avg = self.process_category(cr, scale)



        cmap = 'Reds' 
        # import matplotlib.colors as mcolors
        # from matplotlib.colors import LinearSegmentedColormap
        # Define a monochromatic colormap
        # base_color = 'red'  # You can change this to any color you like
        # base_rgb = mcolors.colorConverter.to_rgb(base_color)
        # cmap = LinearSegmentedColormap.from_list('mono', [(0, base_rgb), (1, (1, 1, 1))], N=256)


        if scale == 4: 
            hit_cases_avg_list = self._reshape_into_org(hit_cases_avg)
            miss_cases_avg_list = self._reshape_into_org(miss_cases_avg)
            fa_cases_avg_list = self._reshape_into_org(fa_cases_avg)
            cr_cases_avg_list = self._reshape_into_org(cr_cases_avg)

        elif scale ==32: 
            hit_cases_avg_list = hit_cases_avg
            miss_cases_avg_list = miss_cases_avg
            fa_cases_avg_list = fa_cases_avg
            cr_cases_avg_list = cr_cases_avg

        if self.embd_type == 'VVT': 
            fig, axs = plt.subplots(1, 4, figsize = (20, 4))

            im = axs[0].imshow(hit_cases_avg_list[0], cmap=cmap)
            plt.colorbar(im, ax=axs[0])
            axs[0].set_title('Hit Cases ViT Explainator')
            axs[0].invert_yaxis()
            contours = axs[0].contour(hit_cases_avg_list[0], colors='black', levels=14)  # You can adjust the number of levels and colors
            axs[0].clabel(contours, inline=True, fontsize=8)


            im = axs[1].imshow(miss_cases_avg_list[0],  cmap=cmap)
            plt.colorbar(im, ax=axs[1])
            axs[1].set_title('Miss Cases ViT Explainator')
            axs[1].invert_yaxis()

            im = axs[2].imshow(fa_cases_avg_list[0],  cmap=cmap)
            plt.colorbar(im, ax=axs[2])
            axs[2].set_title('False Alarm Cases ViT Explainator')
            axs[2].invert_yaxis()

            im = axs[3].imshow(cr_cases_avg_list[0],  cmap=cmap)
            plt.colorbar(im, ax=axs[3])
            axs[3].set_title('Correct Rejection Cases ViT Explainator')
            axs[3].invert_yaxis()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        elif self.embd_type == 'PIT': 
            fig, axs = plt.subplots(5, 4, figsize = (20, 20))

 
            sum_scores = []
            for g in range(5):
                im = axs[g, 0].imshow(hit_cases_avg_list[g], vmin = 0, vmax = 0.25, cmap=cmap) #vmin=0.0, vmax=0.6,
                plt.colorbar(im, ax=axs[g, 0])
                axs[g, 0].set_title(f'G{g+1} Hit Cases [Score = {np.mean(hit_cases_avg_list[g]):.2f}]')
                axs[g, 0].invert_yaxis()


                im = axs[g, 1].imshow(miss_cases_avg_list[g], vmin = 0, vmax = 0.25, cmap=cmap)
                plt.colorbar(im, ax=axs[g, 1])
                axs[g, 1].set_title(f'G{g+1} Miss Cases [Score = {np.mean(miss_cases_avg_list[g]):.2f}]')
                axs[g, 1].invert_yaxis()

                im = axs[g, 2].imshow(fa_cases_avg_list[g], vmin = 0, vmax = 0.25, cmap=cmap)
                plt.colorbar(im, ax=axs[g, 2])
                axs[g, 2].set_title(f'G{g+1} False Alarm Cases [Score = {np.mean(fa_cases_avg_list[g]):.2f}]')
                axs[g, 2].invert_yaxis()


                im = axs[g, 3].imshow(cr_cases_avg_list[g], vmin = 0, vmax = 0.25, cmap=cmap)
                plt.colorbar(im, ax=axs[g, 3])
                axs[g, 3].set_title(f'G{g+1} Correct Rejection Cases [Score = {np.mean(cr_cases_avg_list[g]):.2f}]')
                axs[g, 3].invert_yaxis()

                this_group_score = (np.mean(hit_cases_avg_list[g]) + np.mean(miss_cases_avg_list[g])+ np.mean(fa_cases_avg_list[g]) + np.mean(cr_cases_avg_list[g])) / 4.

                sum_scores.append(this_group_score)
            
            fig.suptitle(f'G1 Score = {sum_scores[0]:.2f} | G2 Score = {sum_scores[1]:.2f} | G3 Score = {sum_scores[2]:.2f} | G4 Score = {sum_scores[3]:.2f} | G5 Score = {sum_scores[4]:.2f}', fontsize = 18)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        elif self.embd_type == 'STT': 
            fig, axs = plt.subplots(4, 4, figsize = (20, 16))

            sum_scores = []
            for g in range(hit_cases_avg.shape[1]):
                im = axs[g, 0].imshow(hit_cases_avg_list[g], vmin = 0, vmax = 0.25, cmap=cmap) #vmin=0.0, vmax=0.6,
                plt.colorbar(im, ax=axs[g, 0])
                #axs[g, 0].set_title(f'T{g+1} Hit Cases [Score = {np.mean(hit_cases_avg_list[g]):.2f}]')
                axs[g, 0].invert_yaxis()
                # contours = axs[g,0].contour(hit_cases_avg_list[0], colors='black', levels=4)  # You can adjust the number of levels and colors
                # axs[g, 0].clabel(contours, inline=True, fontsize=8)


                im = axs[g, 1].imshow(miss_cases_avg_list[g], vmin = 0, vmax = 0.25, cmap=cmap)
                plt.colorbar(im, ax=axs[g, 1])
                #axs[g, 1].set_title(f'T{g+1} Miss Cases [Score = {np.mean(miss_cases_avg_list[g]):.2f}]')
                axs[g, 1].invert_yaxis()


                im = axs[g, 2].imshow(fa_cases_avg_list[g], vmin = 0, vmax = 0.25, cmap=cmap)
                plt.colorbar(im, ax=axs[g, 2])
                #axs[g, 2].set_title(f'T{g+1} False Alarm Cases [Score = {np.mean(fa_cases_avg_list[g]):.2f}]')
                axs[g, 2].invert_yaxis()


                im = axs[g, 3].imshow(cr_cases_avg_list[g], vmin = 0, vmax = 0.25, cmap=cmap)
                plt.colorbar(im, ax=axs[g, 3])
                #axs[g, 3].set_title(f'T{g+1} Correct Rejection Cases [Score = {np.mean(cr_cases_avg_list[g]):.2f}]')
                axs[g, 3].invert_yaxis()

                this_group_score = (np.mean(hit_cases_avg_list[g]) + np.mean(miss_cases_avg_list[g])+ np.mean(fa_cases_avg_list[g]) + np.mean(cr_cases_avg_list[g])) / 4.

                sum_scores.append(this_group_score)
            
            fig.suptitle(f'T1 Score = {sum_scores[0]:.2f} | T2 Score = {sum_scores[1]:.2f} | T3 Score = {sum_scores[2]:.2f} | T4 Score = {sum_scores[3]:.2f}', fontsize = 18)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        elif self.embd_type == 'SVT': 
            fig, axs = plt.subplots(9, 4, figsize = (20, 36))

 
            sum_scores = []
            for g in range(9):
                im = axs[g, 0].imshow(hit_cases_avg_list[g], vmin = 0, vmax = 0.25, cmap=cmap) #vmin=0.0, vmax=0.6,
                plt.colorbar(im, ax=axs[g, 0])
                axs[g, 0].set_title(f'G{g+1} Hit Cases [Score = {np.mean(hit_cases_avg_list[g]):.2f}]')
                axs[g, 0].invert_yaxis()


                im = axs[g, 1].imshow(miss_cases_avg_list[g], vmin = 0, vmax = 0.25, cmap=cmap)
                plt.colorbar(im, ax=axs[g, 1])
                axs[g, 1].set_title(f'G{g+1} Miss Cases [Score = {np.mean(miss_cases_avg_list[g]):.2f}]')
                axs[g, 1].invert_yaxis()

                im = axs[g, 2].imshow(fa_cases_avg_list[g], vmin = 0, vmax = 0.25, cmap=cmap)
                plt.colorbar(im, ax=axs[g, 2])
                axs[g, 2].set_title(f'G{g+1} False Alarm Cases [Score = {np.mean(fa_cases_avg_list[g]):.2f}]')
                axs[g, 2].invert_yaxis()


                im = axs[g, 3].imshow(cr_cases_avg_list[g], vmin = 0, vmax = 0.25, cmap=cmap)
                plt.colorbar(im, ax=axs[g, 3])
                axs[g, 3].set_title(f'G{g+1} Correct Rejection Cases [Score = {np.mean(cr_cases_avg_list[g]):.2f}]')
                axs[g, 3].invert_yaxis()

                this_group_score = (np.mean(hit_cases_avg_list[g]) + np.mean(miss_cases_avg_list[g])+ np.mean(fa_cases_avg_list[g]) + np.mean(cr_cases_avg_list[g])) / 4.

                sum_scores.append(this_group_score)
            
            fig.suptitle(f'{sum_scores[0]:.2f}|{sum_scores[1]:.2f}|{sum_scores[2]:.2f}|{sum_scores[3]:.2f}|{sum_scores[4]:.2f}|{sum_scores[5]:.2f}|{sum_scores[6]:.2f}|{sum_scores[7]:.2f}|{sum_scores[8]:.2f}', fontsize = 18)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def get_predict_cases(self, ):

        hit = []
        miss = []
        fa = []
        cr = []

        # Iterate through each row in the DataFrame
        for index, row in self.df.iterrows():
            if row['ytrue'] == 1 and row['fog_prob'] > self.threshold:
                hit.append(row['date_cycletime'])
            elif row['ytrue'] == 1 and row['fog_prob'] <= self.threshold:
                miss.append(row['date_cycletime'])
            elif row['ytrue'] == 0 and row['fog_prob'] > self.threshold:
                fa.append(row['date_cycletime'])
            else:
                cr.append(row['date_cycletime'])

        return hit, miss, fa, cr

    # Function to process each category
    def process_category(self, dates, scale = int):

        sum_matrix = None
        count = 0
        for date in dates:
            if scale == 4: 
                if date in self.data_dict and 'lpr' in self.data_dict[date]:
                    lpr_matrix = self.data_dict[date]['lpr']
                    if sum_matrix is None:
                        sum_matrix = np.zeros_like(lpr_matrix)
                    sum_matrix += lpr_matrix
                    count += 1
            elif scale == 32: 
                if date in self.data_dict and 'lpr32' in self.data_dict[date]:
                    lpr_matrix = self.data_dict[date]['lpr32']
                    if sum_matrix is None:
                        sum_matrix = np.zeros_like(lpr_matrix)
                    sum_matrix += lpr_matrix
                    count += 1
        average = sum_matrix / count if count > 0 else None
        return average
