#include <algorithm>
#include <cfloat>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */  
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "caffe/layer.hpp"
#include "caffe/layers/softmax_loss_expectation_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void SumWeights(const int nthreads,
          const Dtype* label, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_, const int ignore_location_,
          Dtype* sum_weights) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[s]);
    if (ignore_location_ || (has_ignore_label_ && (label_value == ignore_label_))) { // Unsupervised
			sum_weights[index] = 0;
    } else { // Supervised
			sum_weights[index] = (double)(label[3*spatial_dim + s]) / 255.0;    
    }
  } 
}

template <typename Dtype>
__global__ void SumSupervisedPixels(const int nthreads,
          const Dtype* label, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_, const int ignore_location_, 
          Dtype* num_supervised) {
  CUDA_KERNEL_LOOP(index, nthreads) {
	  const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[s]);
    if (ignore_location_ || (has_ignore_label_ && (label_value == ignore_label_))) { // Unsupervised
      num_supervised[index] = 0;
    } else { // Supervised
	//if ((label_value == 9) || (label_value == 10) || (label_value == 11)){
			num_supervised[index] = 1;
	//}else{
	//	num_supervised[index] = 0;
	//}
    }
  }
}

template <typename Dtype>
__global__ void SoftmaxLossExpectationForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_, 
					const bool ignore_objectness_, const bool ignore_location_, const bool ignore_constraint_,
					const bool ignore_rank_, const bool ignore_classes_, 
					const bool normalize_supervised_, const bool normalize_classes_, 
					const bool normalize_constraint_, const bool normalize_objectness_,
					const bool is_siftflow_, 
					int* max_pixels, int num_supervised, double sum_weights) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		const int s = index % spatial_dim;
		
		const int label_value = static_cast<int>(label[s]);
		const int channels = dim / spatial_dim;

		// Binary vector length 21 (or 33 if SiftFlow) of which classes (including bg) are in image
    bool class_in_image[33];
		for (int i = 0; i < channels; i++) {
			class_in_image[i] = false;
		}
    int L_plus = 0;
    int c = 0;
    
    while (true) { // Gets all the unique numbers out of the 3rd channel
      int class_ = static_cast<int>(label[2*spatial_dim + c]);
			if (is_siftflow_) class_ -= 1; // Need to 0-index classes if SiftFlow
			if ((class_ < 0) && is_siftflow_) break; // Doesn't include "0" (or -1) label if SiftFlow -- unlabelled
			if ((class_ == 0) && !is_siftflow_) break; // Includes background class if PASCAL      
			class_in_image[class_-1] = true;
      
      L_plus++; // Includes background class in count if PASCAL
      //if ((class_ == 0) && !is_siftflow_) break; // Includes background class if PASCAL
      c++;
    }
    int L_minus = channels - L_plus;
		if (L_plus == 0) printf("L_plus is 0\n");

		// Gets the number of supervised vs. unsupervised pixels (0 supervised if image-level-labels)
		int num_unsupervised = spatial_dim;
		num_unsupervised -= num_supervised;

		loss[index] = 0;	
		// Add this term no matter what (whether we have supervised pixels or not)
		// Treat max pixel for the class as supervised
		// Loss "classes in image"	
		if (!ignore_classes_) {
			for (int c = 0; c < channels; c++) {
				// Sums over classes in image (or background) where s is max-scoring pixel for the class
				if (class_in_image[c] && (s == max_pixels[c]-1) ) { 
					double Sc = max(prob_data[c * spatial_dim + s], Dtype(FLT_MIN));
					double classes_term = log(Sc);
					if (normalize_classes_ && (L_plus > 0)) classes_term /= L_plus; // Normalization by number of labels in image
					loss[index] -= classes_term; 
				}
			}
		}
	
		// Loss: supervision
		if (!ignore_location_ && (has_ignore_label_ && (label_value != ignore_label_))) { // Supervised   && ((label_value == 9) || (label_value == 10) || (label_value == 11))
			int local_label_value = label_value;
			if (is_siftflow_) local_label_value--; // 0-index SiftFlow labels (after the check for 255)
			double Sc = max(prob_data[local_label_value * spatial_dim + s], Dtype(FLT_MIN));
			double weight = 1.0;
			
			if (!ignore_rank_) weight = (double)(label[3*spatial_dim + s]) / 255.0;	
			double supervised_term = weight * log(Sc);
			if (normalize_supervised_) {
				if (num_supervised == 0) printf("Num supervised is 0\n");
				if (ignore_rank_ && (num_supervised > 0) ) supervised_term /= num_supervised;
				else supervised_term /= sum_weights;
			}
			loss[index] -= supervised_term;
		}

		// Loss "constraint" for unsupervised (255) pixels
    if (!ignore_constraint_) {
			for (int c = 0; c < channels; c++) {
				// Sums over classes NOT in image where s is max-scoring pixel for the class
				if (!class_in_image[c] && (s == max_pixels[c]-1)) { 
					double Sc = max(prob_data[c * spatial_dim + s], Dtype(FLT_MIN));
					double constraint_term = 0.0;
					if(Sc == 1.0){
						constraint_term = log(0.001);
					}else{					
						constraint_term = log(1-Sc);
					}
					if (normalize_constraint_ && (L_minus > 0)) constraint_term /= L_minus; // Normalization by number of labels NOT in image
					loss[index] -= constraint_term; 
				}
			}   
    }

		// Loss "objectness" for unsupervised (255) pixels 
		if (!ignore_objectness_ && (has_ignore_label_ && (label_value == ignore_label_))) {
			const int objectness = static_cast<int>(label[1 * spatial_dim + s]); // Value between 0 and 255
      const double S0 = max(prob_data[0 * spatial_dim + s], Dtype(FLT_MIN)); // P(class=0) in our model
      const double P = 1.0 - ((double)(objectness+1) / 257.0); // P(class=0) acording to prior
			double objectness_term = 0.0;
			if (S0 == 1.0){
				objectness_term = ((P*log(S0)) + (1-P)*log(0.000001));
			}else{
				objectness_term = ((P*log(S0)) + (1-P)*log(1-S0));
			}
			
			if (normalize_objectness_ && (num_unsupervised > 0)) objectness_term /= num_unsupervised;
			loss[index] -= objectness_term; 
		}
	}
}

template <typename Dtype>
void SoftmaxWithLossExpectationLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int dim = prob_.count() / outer_num_;

	if (!ignore_rank_) {
		CHECK_EQ(outer_num_ * inner_num_ * 4, bottom[1]->count(0))
      << "Number of labels must match the number of predictions because there are four channels,"
      << "one for gt labels per pixel, one for objectness labels per pixel, one for unique classes, " 
			<< "and one for rank weights; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be 4*N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
	} else {
		CHECK_EQ(outer_num_ * inner_num_ * 3, bottom[1]->count(0))
      << "Number of labels must match the number of predictions because there are three channels, "
      << "one for gt labels per pixel and one for objectness labels per pixel and one for unique classes; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be 3*N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
	}

	fflush(stdout);
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = prob_.mutable_gpu_diff();
	const int channels = dim / inner_num_;

	// ------ Get the max-scoring pixels -----------
	float *prob_data_float;
	int num_elements = inner_num_ * channels;
	cudaMalloc((void**)&prob_data_float, sizeof(float) * num_elements);
	
	cublasStatus_t status;
	cublasHandle_t handle;
	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) printf("Cublas create failed\n");

	status = cublasSetVector(num_elements, sizeof(float), prob_data, 1, prob_data_float, 1);
	if (status != CUBLAS_STATUS_SUCCESS) printf("Cublas set vector failed\n");

	int max_pixels[channels]; // Pixel indices for max pixel probability for each class 
	for (int class_ = 0; class_ < channels; class_++) {
    int start_index = class_ * inner_num_;
		int idx_max;
  	status = cublasIsamax(handle, inner_num_, prob_data_float + start_index, 1, &idx_max);
  	if (status != CUBLAS_STATUS_SUCCESS) printf("cublasIsamax failed\n");
		max_pixels[class_] = idx_max;  
  }

	cublasDestroy(handle);
	cudaFree(prob_data_float);
	int *max_pixels1;
	cudaMalloc((void**)&max_pixels1, sizeof(int) * channels);
	status = cublasSetVector(channels, sizeof(int), &max_pixels, 1, max_pixels1, 1);
	// ----------------------------------------------------------------

	// Gets the number of supervised pixels
	SumSupervisedPixels<Dtype><<<CAFFE_GET_BLOCKS(nthreads), 
			CAFFE_CUDA_NUM_THREADS>>>(nthreads, label, inner_num_, has_ignore_label_, ignore_label_, ignore_location_, counts); 
	Dtype num_supervised_d;
	caffe_gpu_asum(nthreads, counts, &num_supervised_d);	
	int num_supervised = (int) num_supervised_d; 

	// Gets the sum of the weights
	Dtype sum_weights;
	if (!ignore_rank_) {
		SumWeights<<<CAFFE_GET_BLOCKS(nthreads), 
				CAFFE_CUDA_NUM_THREADS>>>(nthreads, label, inner_num_, has_ignore_label_, ignore_label_, ignore_location_, counts);
		Dtype sum_weights;
		caffe_gpu_asum(nthreads, counts, &sum_weights);
	} else {
		sum_weights = 0;
	}

  // NOLINT_NEXT_LINE(whitespace/operators)
	SoftmaxLossExpectationForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data,
      outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, ignore_objectness_, ignore_location_, ignore_constraint_, ignore_rank_, ignore_classes_, normalize_supervised_, normalize_classes_, normalize_constraint_, normalize_objectness_, is_siftflow_, max_pixels1, num_supervised, sum_weights); 
	Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  loss /= outer_num_;
	if (loss == 0) printf("Loss is 0\n");
  
	top[0]->mutable_cpu_data()[0] = loss;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void SoftmaxLossExpectationBackwardGPU(const int nthreads, const Dtype* prob_data, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, const bool ignore_objectness_, const bool ignore_location_, 
					const bool ignore_constraint_, const bool ignore_rank_, const bool ignore_classes_, 
					const bool normalize_supervised_, const bool normalize_classes_, 
					const bool normalize_constraint_, const bool normalize_objectness_,
					const bool is_siftflow_, 
					int *max_pixels, int num_supervised, double sum_weights) {

  CUDA_KERNEL_LOOP(index, nthreads) {
		const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[s]);
		const int channels = dim / spatial_dim;

		// Binary vector length 21 (or 33 if SiftFlow) of which classes (including bg) are in image
    bool class_in_image[33];
    for (int i = 0; i < channels; i++) {
      class_in_image[i] = false;
    }
    int L_plus = 0;
    int c = 0;
    
    while (true) { // Gets all the unique numbers out of the 3rd channel
      int class_ = static_cast<int>(label[2*spatial_dim + c]);
      if (is_siftflow_) class_ -= 1; // Need to 0-index classes if SiftFlow
			if ((class_ < 0) && is_siftflow_) break; // Doesn't include "0" (or -1) label if SiftFlow -- unlabelled
      if ((class_ == 0) && !is_siftflow_) break; // Includes background class if PASCAL
      class_in_image[class_-1] = true;
      
      L_plus++; // Includes background class in count if PASCAL
      //if ((class_ == 0) && !is_siftflow_) break; // Includes background class if PASCAL
      c++;
    }
    int L_minus = channels - L_plus;

		 // Gets the number of supervised vs. unsupervised pixels (0 supervised if image-level-labels)
    int num_unsupervised = spatial_dim;
    num_unsupervised -= num_supervised;
    
	double class_weighting[21];
    	class_weighting[0] =  0.0142;
    	class_weighting[1] =  1.1197;
    	class_weighting[2] =  1.3095;
    	class_weighting[3] =  1.1402;
    	class_weighting[4] =  1.5394;
    	class_weighting[5] =  1.8365;
    	class_weighting[6] =  0.7864;
    	class_weighting[7] =  0.5237;
    	class_weighting[8] =  0.3209;
    	class_weighting[9] =  0.8020;
    	class_weighting[10] = 1.6036;
	class_weighting[11] = 0.9457; //ignore_label: 11 ?
    	class_weighting[12] = 0.3601;
    	class_weighting[13] = 1.0876;
    	class_weighting[14] = 0.8660;
    	class_weighting[15] = 0.1353;
    	class_weighting[16] = 1.6485;
    	class_weighting[17] = 1.5983;
    	class_weighting[18] = 0.8211;
    	class_weighting[19] = 0.7461;
    	class_weighting[20] = 1.3046;
    	float sum_class_weighting = 0.0;
	for (int i = 0; i < channels; i++) {
		sum_class_weighting += class_weighting[i];
		
	//  sum_class_weighting_n += (1.0/class_weighting[i]);
    	}

	/*double class_weighting[11];
    	class_weighting[0] =  0.2595;
    	class_weighting[1] =  0.1826;
    	class_weighting[2] =  4.5640;
    	class_weighting[3] =  0.1417;
    	class_weighting[4] =  0.9051;
    	class_weighting[5] =  0.3826;
    	class_weighting[6] =  9.6446;
    	class_weighting[7] =  1.8418;
    	class_weighting[8] =  0.6823;
    	class_weighting[9] =  6.2478;
    	class_weighting[10] =  7.3614;
	float sum_class_weighting = 0.0;
	for (int i = 0; i < channels; i++) {
		sum_class_weighting += class_weighting[i];
		
	//  sum_class_weighting_n += (1.0/class_weighting[i]);
    	}*/

    // Gradient "classes in image"
		// This term was added to loss no matter what
		if (!ignore_classes_) {// && (num_supervised == 0)
			
			int numMaxFor = 0;
			for (int c = 0; c < channels; c++) {
			
	    	// Sums numMaxFor over all classes that are present in image and pixel s is maximal
  	    if (class_in_image[c] && (s == max_pixels[c]-1)) {
    	    numMaxFor++;
     	 	}
    	}
    	for (int c = 0; c < channels; c++) {
		//bottom_diff[c*spatial_dim + s] = prob_data[c * spatial_dim + s];
		/*double Sc = max(prob_data[c * spatial_dim + s], Dtype(FLT_MIN));
		if(numMaxFor > 0){//s es maximo en alguna loncha
			
			if (class_in_image[c]){//class in the image
				if(s == max_pixels[c]-1) { // s is max in loncha c
					bottom_diff[c*spatial_dim + s] = Sc - 1;
				}else{ // s is not max in loncha c
					
					bottom_diff[c*spatial_dim + s] = Sc;
				}
			/*printf("grad: %f %f %d %d %d\n", bottom_diff[c*spatial_dim + s], prob_data[c * spatial_dim + s], numMaxFor, s, c);
			double sum= 0.0;
			double sum2= 0.0;
			for (int cdb = 0; cdb < channels; cdb++) {
				sum += bottom_diff[cdb*spatial_dim + s];
				sum2 += prob_data[cdb * spatial_dim + s];
				printf("bucle: %f %f %d\n", bottom_diff[cdb*spatial_dim + s], prob_data[cdb * spatial_dim + s], cdb);
			}
			printf("suma: %f %f\n", sum, sum2);
			}else{//no class => no evaluation or pass Sc?
				bottom_diff[c*spatial_dim + s] = Sc;
			}
		//if(numMaxFor > 0){ 
		//	double Sc = max(prob_data[c * spatial_dim + s], Dtype(FLT_MIN));
    		//	bottom_diff[c*spatial_dim + s] = Sc;//*= numMaxFor;
		}else{*/
		bottom_diff[c*spatial_dim + s] *= numMaxFor;
		//}
		if (class_in_image[c] && (s == max_pixels[c]-1)) {
			//printf("numMaxFor: %d\n", numMaxFor);
			//double Sc = max(prob_data[c * spatial_dim + s], Dtype(FLT_MIN));
			//double class_term;
			//class_term = Sc - 1;
        		//bottom_diff[c*spatial_dim + s] = class_term;//-= 1; //SOFTMAX
			bottom_diff[c*spatial_dim + s] -= 1; //SOFTMAX
      		}
				// Normalize by number of classes in image
				if (normalize_classes_ && (L_plus > 0)) bottom_diff[c*spatial_dim + s] /= L_plus; 
    	}
		
	} else { //ignore_classes: true
			for (int c = 0; c < channels; c++) {
				bottom_diff[c*spatial_dim + s] = 0;
			}
	}		

	

		//Gradient: supervision
    if (!ignore_location_ && (has_ignore_label_ && (label_value != ignore_label_))) { // Supervised 
			int local_label_value = label_value;
			//if (!class_flag[local_label_value]){
			//	class_flag[local_label_value] = 1;
			// Calculate R = sum of S_ic / (1-S_ic)
			double R = 0;
			for (int c = 0; c < channels; c++) {
				if (!class_in_image[c]) {
					double Sc = max(prob_data[c * spatial_dim + s], Dtype(FLT_MIN));
					if (Sc == 1.0){
						R += (Sc / (1-0.999));
					}else{
						R += (Sc / (1-Sc));
					}
					
				}
			}
      if (is_siftflow_) local_label_value--; // 0-index SiftFlow labels (after the check for 255)
			double weight = 1.0;
			//weight = class_weighting[local_label_value];
      if (!ignore_rank_) weight = (double)(label[3*spatial_dim + s]) / 255.0;
			for (int c = 0; c < channels; c++) {

				double Sc = max(prob_data[c * spatial_dim + s], Dtype(FLT_MIN));
				double supervised_term;
				
				//if (class_in_image[c]){
					if (c == local_label_value){// && class_in_image[c]) { // For supervised pixel, for target class t, gradient is S_t - 1
						supervised_term = weight * (Sc - 1);
					//} else { // For supervised pixel, for any other class, gradient is S_t
					//	supervised_term = weight * Sc;
					//}
				} else { // For supervised pixel, for any other class, gradient is S_t
					/*if (!class_in_image[c]){
						if (Sc == 1.0){ 
							supervised_term = weight * 1.0; //-Sc*R + (Sc / (1-0.999)); --> Esto da 0 y las perdidas son maximas! 
						}else{
							supervised_term = weight * (-Sc*R + (Sc / (1-Sc)));
						}
					}else{*/
						supervised_term = weight * Sc;
					//}
				}
				//	if (normalize_classes_ && (L_plus > 0)) supervised_term /= L_plus;
				/*}else{
					if (c == local_label_value) { // For supervised pixel, for target class t, gradient is S_t - 1
						if (Sc == 1.0){ 
							supervised_term = 1.0; //-Sc*R + (Sc / (1-0.999)); --> Esto da 0 y las perdidas son maximas! 
						}else{
							supervised_term = -Sc*R + (Sc / (1-Sc));
						}
						printf("Hola : %f %f %f\n", supervised_term, Sc, R);
					} else { // For supervised pixel, for any other class, gradient is S_t
						supervised_term = -Sc*R;
						printf("Hola 2: %f %f %f\n", supervised_term, Sc, R);
					}
					if (normalize_constraint_ && (L_minus > 0)) supervised_term /= L_minus;
				}*/
				//
				if (normalize_supervised_) {
					//supervised_term /= sum_class_weighting;
        	if (ignore_rank_ && (num_supervised > 0)) supervised_term /= num_supervised;
        	else supervised_term /= sum_weights;
      	}
		/*if (supervised_term == 0.0){
			printf("supervised_term: %f %f %f\n", supervised_term, Sc, R);
			printf("supervised_term: %d %d %d\n", c, local_label_value, class_in_image[c]);
		}*/
        bottom_diff[c*spatial_dim + s] = supervised_term;
			} 
		} // For unsupervised pixel, gradient is 0 for all classes (already set)
//}//if
		// Gradient "constraint" for unsupervised (255) pixels
    
    if (!ignore_constraint_) {// && (num_supervised == 0)
			// Calculate R = sum of S_ic / (1-S_ic)
			double R = 0;
			for (int c = 0; c < channels; c++) {
				if (!class_in_image[c] && (s == max_pixels[c]-1) ) {
					double Sc = max(prob_data[c * spatial_dim + s], Dtype(FLT_MIN));
					if (Sc == 1.0){
						R += (Sc / (1-0.999));
					}else{
						R += (Sc / (1-Sc));
					}
					
				}
			}
			/*printf("pixel: %d\n", s);
			
			printf("R: %f\n", R);
			printf("channels: %d\n", channels);
			for(int db = 0; db < channels; db++){
				//printf("max_pixels[c]-1: %d\n", max_pixels[db]-1);
				if (class_in_image[db]){
					printf("class: %d\n", db);
				}
			}*/

      for (int c = 0; c < channels; c++) {
        double Sc = max(prob_data[c * spatial_dim + s], Dtype(FLT_MIN));
				double constraint_term = 0.0;

        if (!class_in_image[c] && (s == max_pixels[c]-1)) {
				if (Sc == 1.0){ 
					constraint_term = 1.0; //-Sc*R + (Sc / (1-0.999)); --> Esto da 0 y las perdidas son maximas! 
				}else{
					constraint_term = -Sc*R + (Sc / (1-Sc));
				}
		        /*if (Sc == 1.0){
        			printf("ERROR max pixel %f %f\n", R, constraint_term);
        		}*/
        } else {
					constraint_term = -Sc*R;

			
        }
				if (normalize_constraint_ && (L_minus > 0)) constraint_term /= L_minus;
	//printf("constraint: %f\n", constraint_term);
        bottom_diff[c*spatial_dim + s] += constraint_term;
      }
    }

		// Gradient "objectness" for unsupervised (255) pixels
		if (!ignore_objectness_ && (has_ignore_label_ && (label_value == ignore_label_))) {
			const int objectness = static_cast<int>(label[1 * spatial_dim + s]); // Value between 0 and 255
  	  const double S0 = max(prob_data[0 * spatial_dim + s], Dtype(FLT_MIN)); // P(class=0) in our model
	    const double P = 1.0 - ((double)(objectness+1) / 257.0); // P(class=0) acording to prior
		
			for (int c = 0; c < channels; c++) {
				double objectness_term;
				if (c == 0) { // Background
					objectness_term = S0 - P;
				} else { 
					double Sc = max(prob_data[c*spatial_dim + s], Dtype(FLT_MIN));
					if (S0 == 1.0){
						objectness_term = Sc*((P-S0)/(0.000001));
					}else{
						objectness_term = Sc*((P-S0)/(1-S0));
					}		
				}
				if (normalize_objectness_ && (num_unsupervised > 0)) objectness_term /= num_unsupervised;
				bottom_diff[c*spatial_dim + s] += objectness_term;
			}
			// For supervised pixels, gradient objectness is 0 (already done)
		} 

	}
}

template <typename Dtype>
void SoftmaxWithLossExpectationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();
		const int channels = dim / inner_num_;

		// ------ Get the max-scoring pixels -----------
  	float *prob_data_float;
  	int num_elements = inner_num_ * channels;
  	cudaMalloc((void**)&prob_data_float, sizeof(float) * num_elements);


  	cublasStatus_t status;
  	cublasHandle_t handle;
  	status = cublasCreate(&handle);
  	if (status != CUBLAS_STATUS_SUCCESS) printf("Cublas create failed\n");

  	status = cublasSetVector(num_elements, sizeof(float), prob_data, 1, prob_data_float, 1);
  	if (status != CUBLAS_STATUS_SUCCESS) printf("Cublas set vector failed\n");

  	int max_pixels[channels]; // Pixel indices for max pixel probability for each class 
  	for (int class_ = 0; class_ < channels; class_++) {
    	int start_index = class_ * inner_num_;
    	int idx_max;
    	status = cublasIsamax(handle, inner_num_, prob_data_float + start_index, 1, &idx_max);
    	if (status != CUBLAS_STATUS_SUCCESS) printf("cublasIsamax failed\n");
    	max_pixels[class_] = idx_max;
  	}

  	cublasDestroy(handle);
  	cudaFree(prob_data_float);
  	int *max_pixels1;
  	cudaMalloc((void**)&max_pixels1, sizeof(int) * channels);
	  status = cublasSetVector(channels, sizeof(int), &max_pixels, 1, max_pixels1, 1);
		// --------------------------------------------------------

		// Gets the number of supervised pixels
	  SumSupervisedPixels<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
  	    CAFFE_CUDA_NUM_THREADS>>>(nthreads, label, inner_num_, has_ignore_label_, ignore_label_, ignore_location_, counts);
  	Dtype num_supervised_d;
  	caffe_gpu_asum(nthreads, counts, &num_supervised_d);
		int num_supervised = (int) num_supervised_d;

		// Gets the sum of the weights
		Dtype sum_weights;
  	if (!ignore_rank_) {
			SumWeights<<<CAFFE_GET_BLOCKS(nthreads),
    	  	CAFFE_CUDA_NUM_THREADS>>>(nthreads, label, inner_num_, has_ignore_label_, ignore_label_, ignore_location_, counts);
  		caffe_gpu_asum(nthreads, counts, &sum_weights);
		} else {
			sum_weights = 0;
		}

    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxLossExpectationBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, top_data, label, bottom_diff,
        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, ignore_objectness_, ignore_location_, ignore_constraint_, ignore_rank_, ignore_classes_, normalize_supervised_, normalize_classes_, normalize_constraint_, normalize_objectness_, is_siftflow_, max_pixels1, num_supervised, sum_weights);
  
	  const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossExpectationLayer);

}  // namespace caffe
