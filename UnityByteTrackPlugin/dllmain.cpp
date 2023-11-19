#include "pch.h"
#include <string>
#include <vector>
#include <functional>

#include <Eigen/Dense>
#include "BYTETracker.h"

#define DLLExport __declspec (dllexport)

extern "C" {

	BYTETracker* tracker = nullptr;
	const int BOX_PROBS_COLS = 5;

	DLLExport void init_tracker(float track_thresh=0.23f, int track_buffer=30, float match_thresh=0.8f, int frame_rate = 30) {
		tracker = new BYTETracker(track_thresh, track_buffer, match_thresh, frame_rate);
	}

	DLLExport void get_track_ids(float* boxes_probs_array, int num_detections, int* track_ids_array) {

		Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> boxes_probs_map(boxes_probs_array, num_detections, BOX_PROBS_COLS);

		Eigen::MatrixXf boxes_probs(boxes_probs_map);

		std::vector<KalmanBBoxTrack>tracks = tracker->process_frame_detections(boxes_probs);

		Eigen::MatrixXf tlbr_boxes(boxes_probs.rows(), 4);
		tlbr_boxes << boxes_probs.col(0),          // top-left X
			boxes_probs.col(1),                    // top-left Y
			boxes_probs.col(0) + boxes_probs.col(2), // bottom-right X
			boxes_probs.col(1) + boxes_probs.col(3); // bottom-right Y
		
		std::vector<int> track_ids = match_detections_with_tracks(tlbr_boxes.cast<double>(), tracks);

		// Copy the track_ids to the track_ids_array
		std::memcpy(track_ids_array, track_ids.data(), num_detections * sizeof(int));
	}
}