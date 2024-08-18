import json
import os

import gradio as gr
import pandas as pd
import requests

# Define the URL for the FastAPI service
BASE_URL = os.environ.get("API_BASE_URL", default="http://app:8000")


def upload_csv(file):
    filename = file.name
    with open(filename, 'rb') as f:
        files = {'file': (filename, f, 'text/csv')}
        response = requests.post(f"{BASE_URL}/upload-csv/", files=files)
        if response.status_code == 200:
            return response.json()['message']
        else:
            return response.json()['detail']


def search_names(query, k=5):
    if query == "":
        raise gr.Error("Please enter a search query")

    if k < 1:
        raise gr.Error("Number of results should be at least 1")

    response = requests.get(f"{BASE_URL}/search/", params={'query': query, 'k': k})
    if response.status_code == 200:
        data = response.json()
        semantic_results = data['match']['semantic']
        typo_results = data['match']['typo']

        semantic_df = pd.DataFrame(semantic_results)
        typo_df = pd.DataFrame(typo_results)

        return semantic_df, typo_df
    else:
        return response.json()['detail'], pd.DataFrame()


def delete_collection():
    response = requests.delete(f"{BASE_URL}/purge/")
    if response.status_code == 200:
        return response.json()['message']
    else:
        return response.json()['detail']


def check_task_status():
    response = requests.get(f"{BASE_URL}/task-status/")
    if response.status_code == 200:
        resp_str = json.dumps(response.json(), indent=4)
        return resp_str
    else:
        return "Something bad happened"


def add_override(tag, synonyms):
    synonyms_list = [syn.strip() for syn in synonyms.split(',')]
    data = {"tag": tag, "synonyms": synonyms_list}
    response = requests.post(f"{BASE_URL}/overrides/", json=data)
    if response.status_code == 200:
        return response.json()['message']
    else:
        return response.json()['detail']


def delete_override(tag):
    response = requests.delete(f"{BASE_URL}/overrides/{tag}")
    if response.status_code == 200:
        return response.json()['message']
    else:
        return response.json()['detail']


def get_overrides():
    response = requests.get(f"{BASE_URL}/overrides/")
    if response.status_code == 200:
        overrides = response.json()['manual_overrides']
        overrides_df = pd.DataFrame([{'Tag': tag, 'Synonyms': ', '.join(synonyms)}
                                    for tag, synonyms in overrides.items()])
        return overrides_df
    else:
        return pd.DataFrame()


# Gradio interface
with gr.Blocks(css="body { font-family: Arial, sans-serif; } footer { visibility: hidden }") as demo:
    gr.Markdown(
        "<h1 style='text-align: center;'>Tag Search API</h1><h3 style='text-align: center;'>by the AI Team</h3>")

    with gr.Tab("Search"):
        with gr.Row():
            query = gr.Textbox(label="Search Query",
                               placeholder="Enter tag to search", elem_classes="tab-container")
            k = gr.Number(label="Number of Results", value=5, elem_classes="tab-container")
        search_button = gr.Button("Search", elem_classes="tab-container")
        semantic_table = gr.Dataframe(label="Semantic Matches",
                                      interactive=False, elem_classes="tab-container")
        typo_table = gr.Dataframe(label="Typo Matches", interactive=False,
                                  elem_classes="tab-container")

        search_button.click(fn=search_names, inputs=[query, k], outputs=[
                            semantic_table, typo_table])
        query.submit(fn=search_names, inputs=[query, k], outputs=[semantic_table, typo_table])

    with gr.Tab("Upload CSV"):
        with gr.Accordion("Information", open=False):
            gr.Markdown("""## Info
                        - the csv file must be **comma separated**
                        - the csv file **should have a 'name' column**, anything else will be discarded
                        - processing time depends on the number of records, you can check the status with the button below
                        """)
        csv_file = gr.File(label="Upload CSV File", type='filepath', elem_classes="tab-container")
        upload_button = gr.Button("Upload", elem_classes="tab-container")
        upload_status = gr.Textbox(label="Upload Status", interactive=False,
                                   elem_classes="tab-container")

        upload_button.click(fn=upload_csv, inputs=csv_file, outputs=upload_status)

        status_button_upload = gr.Button("Check Task Status", elem_classes="tab-container")
        status_display_upload = gr.Textbox(
            label="Task Status", interactive=False, elem_classes="tab-container")
        status_button_upload.click(fn=check_task_status, outputs=status_display_upload)

        delete_button = gr.Button("Delete Collection", elem_classes="tab-container")
        delete_status = gr.Textbox(label="Delete Status", interactive=False,
                                   elem_classes="tab-container")
        delete_button.click(fn=delete_collection, outputs=delete_status)

    with gr.Tab("Manage Overrides"):
        with gr.Row():
            tag = gr.Textbox(label="Tag", placeholder="Enter the tag to override",
                             elem_classes="tab-container")
            synonyms = gr.Textbox(
                label="Synonyms", placeholder="Enter synonyms separated by commas", elem_classes="tab-container")
        add_override_button = gr.Button("Add Override", elem_classes="tab-container")
        add_override_status = gr.Textbox(
            label="Status", interactive=False, elem_classes="tab-container")

        add_override_button.click(fn=add_override, inputs=[
                                  tag, synonyms], outputs=add_override_status)

        with gr.Row():
            delete_tag = gr.Textbox(
                label="Tag", placeholder="Enter the tag to delete override", elem_classes="tab-container")
        delete_override_button = gr.Button("Delete Override", elem_classes="tab-container")
        delete_override_status = gr.Textbox(
            label="Delete Status", interactive=False, elem_classes="tab-container")

        delete_override_button.click(fn=delete_override, inputs=delete_tag,
                                     outputs=delete_override_status)

        view_overrides_button = gr.Button("View All Overrides", elem_classes="tab-container")
        overrides_table = gr.Dataframe(label="Current Overrides",
                                       interactive=False, elem_classes="tab-container")

        view_overrides_button.click(fn=get_overrides, outputs=overrides_table)

# Launch the interface
demo.launch(server_name="0.0.0.0")
