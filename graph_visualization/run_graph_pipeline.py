import os

input_files = open("/nfs/home/cheemag/fake_narratives/video_list.txt", "r").readlines()

base_input_path = "/nfs/data/fakenarratives/202306_corpus/results_pkl/"
base_output_path = "/nfs/data/fakenarratives/202306_corpus/20231122_results_graphml/"

for i, input_file in enumerate(input_files):

    print(f"Processing video {i}/{len(input_files)}: {input_file}")

    input_path = os.path.join(base_input_path, input_file.strip())
    output_path = os.path.join(base_output_path, input_file.split("/")[0])

    command = f"python graph_visualization/export_graphml.py -i {input_path} -c graph_visualization/config.yml -o {output_path} -a whisper --pyviz"

    os.system(command)

    command = f"python graph_visualization/export_graphml.py -i {input_path} -c graph_visualization/config.yml -o {output_path} -a whisperx --pyviz"

    os.system(command)

    print()
