#include "MotionTracker.h"
#include <fstream>

MotionTracker::MotionTracker(int frame_rate, int track_buffer)
{
	track_thresh = 0.5;//（yolov7-w6(1920):0.6;yolov7-tiny(1920):0.5;yolov7-tiny(640):0.4;yolov7-tiny(960):0.5）
	det_thresh = 0.6;//（yolov7-w6(1920):0.7;yolov7-tiny(1920):0.6;yolov7-tiny(640):0.5;yolov7-tiny(960):0.6）
	motion_thresh = 180;

	frame_id = 0;
	max_time_lost = int(frame_rate / 30.0 * track_buffer);

	cout << "Init MotionTrack!" << endl;
}

MotionTracker::~MotionTracker()
{
}

vector<STrack> MotionTracker::update(const vector<Detection>& objects)
{
	if(this->frame_id == 0)
		STrack::init_id();

	////////////////// Step 1: Get detections //////////////////
	this->frame_id++;
	vector<STrack> activated_stracks;
	vector<STrack> refind_stracks;
	vector<STrack> removed_stracks;
	vector<STrack> lost_stracks;
	vector<STrack> detections;
	vector<STrack> detections_low;

	vector<STrack> tracked_stracks_swap;
	vector<STrack> resa, resb;
	vector<STrack> output_stracks;

	vector<STrack*> unconfirmed;
	vector<STrack*> tracked_stracks;
	vector<STrack*> strack_pool;

	if (objects.size() > 0)
	{
		for (int i = 0; i < objects.size(); i++)
		{
			vector<float> tlbr_;
			tlbr_.resize(4);
			tlbr_[0] = objects[i].box.x;
			tlbr_[1] = objects[i].box.y;
			tlbr_[2] = objects[i].box.x + objects[i].box.width;
			tlbr_[3] = objects[i].box.y + objects[i].box.height;
			float score = objects[i].conf;
			int classes = objects[i].classId;
			STrack strack(STrack::tlbr_to_tlwh(tlbr_), score,classes);
			if (score >= track_thresh)
				detections.push_back(strack);
			else
				detections_low.push_back(strack);
		}
	}

	// Add newly detected tracklets to tracked_stracks
	for (int i = 0; i < this->tracked_stracks.size(); i++)
	{
		if (!this->tracked_stracks[i].is_activated)
			unconfirmed.push_back(&this->tracked_stracks[i]);
		else
			tracked_stracks.push_back(&this->tracked_stracks[i]);
	}

	////////////////// Step 2: First association, with IoU distance //////////////////
	strack_pool = joint_stracks(tracked_stracks, this->lost_stracks);
	STrack::multi_predict(strack_pool, this->kalman_filter);
	vector<vector<float> > dists;
	int dist_size = 0, dist_size_size = 0;
	dists = iou_distance(strack_pool, detections, dist_size, dist_size_size);
	fuse_classes_width_height(dists,strack_pool,detections);
	vector<vector<int> > matches;
	vector<int> u_track, u_detection;
	linear_assignment(dists, dist_size, dist_size_size, 0.98, matches, u_track, u_detection);
	for (int i = 0; i < matches.size(); i++)
	{
		STrack *track = strack_pool[matches[i][0]];
		STrack *det = &detections[matches[i][1]];
		if (track->state == TrackState::Tracked)
		{
			track->update(*det, this->frame_id);
			activated_stracks.push_back(*track);
		}
		else
		{
			track->re_activate(*det, this->frame_id, false);
			refind_stracks.push_back(*track);
		}
	}

	index_detection(detections,u_detection);

	////////////////// Step 3: Second association, using low score dets //////////////////

	index_strack_pool(strack_pool,u_track);
	dists.clear();
	dists = iou_distance(strack_pool, detections_low, dist_size, dist_size_size);
	fuse_classes_width_height(dists,strack_pool,detections_low);
	matches.clear();
	u_track.clear();
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 0.98, matches, u_track, u_detection);
	for (int i = 0; i < matches.size(); i++)
	{
		STrack *track = strack_pool[matches[i][0]];
		STrack *det = &detections_low[matches[i][1]];
		if (track->state == TrackState::Tracked)
		{
			track->update(*det, this->frame_id);
			activated_stracks.push_back(*track);
		}
		else
		{
			track->re_activate(*det, this->frame_id, false);
			refind_stracks.push_back(*track);
		}
	}

	// index_detection(detections_low,u_detection);
	detections_low.clear();
	////////////////// Step 4: Thrid association, using gaussian distance //////////////////

	index_strack_pool(strack_pool,u_track);
	dists.clear();
	dists = motion_distance(strack_pool, detections,this->motion_thresh,dist_size, dist_size_size);
	fuse_classes_width_height(dists,strack_pool,detections);
	matches.clear();
	u_track.clear();
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 0.98, matches, u_track, u_detection);
	for (int i = 0; i < matches.size(); i++)
	{
		STrack *track = strack_pool[matches[i][0]];
		STrack *det = &detections[matches[i][1]];
		if (track->state == TrackState::Tracked)
		{
			track->update(*det, this->frame_id);
			activated_stracks.push_back(*track);
		}
		else
		{
			track->re_activate(*det, this->frame_id, false);
			refind_stracks.push_back(*track);
		}
	}
	////////////////// Step 5: Mark lost stracks //////////////////
	for (int i = 0; i < u_track.size(); i++)
	{
		STrack *track = strack_pool[u_track[i]];
		if (track->state != TrackState::Lost)
		{
			track->mark_lost();
			lost_stracks.push_back(*track);
		}
	}
	////////////////// Step 6:Deal with unconfirmed tracks using IoU distance//////////////////
	index_detection(detections,u_detection);
	dists.clear();
	dists = iou_distance(unconfirmed, detections, dist_size, dist_size_size);
	fuse_classes_width_height(dists,unconfirmed,detections);
	matches.clear();
	vector<int> u_unconfirmed;
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 0.98, matches, u_unconfirmed, u_detection);

	for (int i = 0; i < matches.size(); i++)
	{
		unconfirmed[matches[i][0]]->update(detections[matches[i][1]], this->frame_id);
		activated_stracks.push_back(*unconfirmed[matches[i][0]]);
	}
	
	////////////////// Step 7: Deal with unconfirmed tracks using gaussian distance//////////////////
	index_detection(detections,u_detection);
	index_strack_pool(unconfirmed,u_unconfirmed);
	dists.clear();
	dists = motion_distance(unconfirmed, detections,this->motion_thresh, dist_size, dist_size_size);
	fuse_classes_width_height(dists,unconfirmed,detections);
	matches.clear();
	u_unconfirmed.clear();
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 0.98, matches, u_unconfirmed, u_detection);

	for (int i = 0; i < matches.size(); i++)
	{
		unconfirmed[matches[i][0]]->update(detections[matches[i][1]], this->frame_id);
		activated_stracks.push_back(*unconfirmed[matches[i][0]]);
	}

	////////////////// Step 8: Remove unconfirmed stracks //////////////////
	for (int i = 0; i < u_unconfirmed.size(); i++)
	{
		STrack *track = unconfirmed[u_unconfirmed[i]];
		track->mark_removed();
		removed_stracks.push_back(*track);
	}

	////////////////// Step 9:  Init new stracks //////////////////
	for (int i = 0; i < u_detection.size(); i++)
	{
		STrack *track = &detections[u_detection[i]];
		if (track->score < this->det_thresh)
			continue;
		track->activate(this->kalman_filter, this->frame_id);
		activated_stracks.push_back(*track);
	}

	////////////////// Step 10: Update state //////////////////
	for (int i = 0; i < this->lost_stracks.size(); i++)
	{
		if (this->frame_id - this->lost_stracks[i].end_frame() > this->max_time_lost)
		{
			this->lost_stracks[i].mark_removed();
			removed_stracks.push_back(this->lost_stracks[i]);
		}
	}
	
	for (int i = 0; i < this->tracked_stracks.size(); i++)
	{
		if (this->tracked_stracks[i].state == TrackState::Tracked)
		{
			tracked_stracks_swap.push_back(this->tracked_stracks[i]);
		}
	}
	this->tracked_stracks.clear();
	this->tracked_stracks.assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

	this->tracked_stracks = joint_stracks(this->tracked_stracks, activated_stracks);
	this->tracked_stracks = joint_stracks(this->tracked_stracks, refind_stracks);

	//std::cout << activated_stracks.size() << std::endl;

	this->lost_stracks = sub_stracks(this->lost_stracks, this->tracked_stracks);
	for (int i = 0; i < lost_stracks.size(); i++)
	{
		this->lost_stracks.push_back(lost_stracks[i]);
	}

	this->lost_stracks = sub_stracks(this->lost_stracks, this->removed_stracks);
	for (int i = 0; i < removed_stracks.size(); i++)
	{
		this->removed_stracks.push_back(removed_stracks[i]);
	}
	
	remove_duplicate_stracks(resa, resb, this->tracked_stracks, this->lost_stracks);

	this->tracked_stracks.clear();
	this->tracked_stracks.assign(resa.begin(), resa.end());
	this->lost_stracks.clear();
	this->lost_stracks.assign(resb.begin(), resb.end());
	
	for (int i = 0; i < this->tracked_stracks.size(); i++)
	{
		if (this->tracked_stracks[i].is_activated)
		{
			output_stracks.push_back(this->tracked_stracks[i]);
		}
	}
	return output_stracks;
}
