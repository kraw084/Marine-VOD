import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

from .Eval_utils import Evaluator

def metric_by_frame_graph(video, metric_name, metric_values):
    """Create a graph to display how a given metric changes throughout the video"""
    plt.plot(range(video.num_of_frames), metric_values, color="red")
    plt.title(f"{metric_name} by frame - {video.name}")
    plt.xlabel("Frame")
    plt.ylabel(metric_name)
    
    plt.show()
    
    
def mt_heatmap(video, gt_tracklets, tracklet_sets, tracker_names):
    gt_ids = [gt_tracklet.id for gt_tracklet in gt_tracklets]
    tracked_status = []
    for ts, tracker_name in zip(tracklet_sets, tracker_names):
        evaluator = Evaluator(tracker_name)
        evaluator.set_tracklets(gt_tracklets, ts)
        evaluator.eval_video()
        tracked_proportions = evaluator.compute_gt_track_status()
                
        tracked_status.append(tracked_proportions)
        
    matplotlib.use('TkAgg')
    sns.heatmap(tracked_status, cmap="YlGnBu", xticklabels=gt_ids, yticklabels=tracker_names)
    plt.xlabel("Gt ID")
    plt.ylabel("Tracker Names")
    plt.title(f"Heatmap of Tracked Status in {video.name} by Tracker")
    plt.xticks(fontsize=8)
    plt.tight_layout()
    plt.show()


def tracklet_trail_graph(gt_tracklet, pred_tracklets, tracker_names, video):
    gt_points = [(box[0], box[1]) for _, box in gt_tracklet]
    pred_points = [[(box[0], box[1]) for _, box in t] for t in pred_tracklets]

    w, h = video.size

    matplotlib.use('TkAgg')

    x, y = [p[0] for p in gt_points], [p[1] for p in gt_points]
    plt.plot(x, y, label=f"GT ({len(x)})", color="green")

    colours = ["red", "blue", "cyan", "magenta", "yellow"][:len(tracker_names)]
    for points, name, colour in zip(pred_points, tracker_names, colours):
        x, y = [p[0] for p in points], [p[1] for p in points]
        plt.plot(x, y, label=f"{name} ({len(x)})", color=colour)

    plt.xlim(0, w)
    plt.ylim(h, 0)
    plt.legend()
    plt.title(f"Trail of gt {gt_tracklet.id} in video {video.name}")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.show()
