# SPDX-FileCopyrightText: 2025 Meinhard Kissich
# SPDX-License-Identifier: MIT

import os
import streamlit as st
import configparser
from transformers import AutoTokenizer, T5ForConditionalGeneration
from contextlib import nullcontext
from annotated_text import annotated_text
from difflib import SequenceMatcher


class App():
    def __init__(self):
        self.config_path = "checkr.ini"
        self.config_defaults_path = "checkr_defaults.ini"
        st.session_state["generated_text"] = ""
        st.session_state["action"] = "grammar"
        st.session_state["should_save"] = False
        self.load_config()
        st.set_page_config(page_title="Checkr", layout="wide")

    def load_config(self, defaults=False):
        parser = configparser.ConfigParser()
        parser.read(self.config_path if not defaults else self.config_defaults_path)
        st.session_state["config"] = {section: dict(parser.items(section)) for section in parser.sections()}
        if defaults:
            self.store_config(False)

    def __should_save(self):
        st.session_state["should_save"] = True

    def __update_to_config_dict(self):
        st.session_state["config"]["LLM"]["action"] = str(st.session_state.get(f"LLM_action"))
        st.session_state["config"]["LLM"]["model"] = str(st.session_state.get(f"LLM_model"))
        st.session_state["config"]["LLM"]["max_length"] = str(st.session_state.get(f"LLM_max_length"))
        st.session_state["config"]["UI"]["show_spinner"] = str(st.session_state.get(f"UI_show_spinner"))
        for k in st.session_state["config"]["PROMPTS"].keys():
            st.session_state["config"]["PROMPTS"][k] = st.session_state.get(f"PROMPTS_{k}")
            st.session_state["config"]["DIFFS"][k] = str(st.session_state.get(f"DIFFS_{k}"))
    
    def store_config(self, from_ui:bool=True):
        if from_ui:
            self.__update_to_config_dict()
        parser = configparser.ConfigParser()
        for section in st.session_state["config"].keys():
            parser.add_section(section)

        for section in st.session_state["config"].keys():
            section_dict = st.session_state["config"][section]
            fields = section_dict.keys()
            for field in fields:
                value = section_dict[field]
                parser.set(section, field, str(value))

        if self.config_path is not None:
            with open(self.config_path, 'w') as f:
                parser.write(f)
        st.session_state["should_save"] = False


    @st.dialog("Update Model & Prompts", width="large")
    def model_prompt_setup(self):
        st.text_input(
            "Model",
            st.session_state["config"]["LLM"]["model"],
            key=f"LLM_model",
            on_change=self.__should_save)

        st.text_input(
            "Max Length",
            st.session_state["config"]["LLM"]["max_length"],
            key=f"LLM_max_length",
            on_change=self.__should_save)

        for k, v in st.session_state["config"]["PROMPTS"].items():
            st.text_input(
                k.title(),
                v,
                key=f"PROMPTS_{k}",
                on_change=self.__should_save)

            st.toggle(
                f"Show Diff for {k}",
                value=st.session_state["config"]["DIFFS"][k] == "True",
                key=f"DIFFS_{k}",
                on_change=self.__should_save)

        st.toggle(
            "Show Spinner while generating",
            key="UI_show_spinner",
            value=st.session_state["config"]["UI"]["show_spinner"] == "True",
            on_change=self.__should_save)

        if st.session_state["should_save"]:
            self.store_config()

    def annotate(self, original_txt, new_txt):
        fg = "#283816"
        original_tokens = original_txt.strip().split()
        modified_tokens = new_txt.strip().split()
        matcher = SequenceMatcher(None, original_tokens, modified_tokens)
        annotated_list = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                annotated_list.append((" ".join(modified_tokens[j1:j2]), "", fg))
            elif tag == 'delete':
                annotated_list.append(("", ""))
            elif tag == 'insert':
                annotated_list.append((" ".join(modified_tokens[j1:j2]), "", fg))
            else:
                annotated_list.append(" ".join(modified_tokens[j1:j2]))
        return annotated_list

    def run_llm(self):
        with st.spinner(r'Downloading Model (very slow on first time! ~5-20min) / Processing. Please be patient...') if \
            st.session_state["config"]["UI"]["show_spinner"] == "True" else nullcontext():
            try:
                tokenizer = AutoTokenizer.from_pretrained(st.session_state["config"]["LLM"]["model"])
                model = T5ForConditionalGeneration.from_pretrained(st.session_state["config"]["LLM"]["model"])
            except Exception as e:
                st.error(f"Cannot load model. Check your settings! {e}")
                return
            
            input_text = st.session_state["config"]["PROMPTS"][str(st.session_state["action"]).lower()] + \
                            " " + st.session_state.get(f"main_input")
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids

            try:
                max_len = int(st.session_state["config"]["LLM"]["max_length"])
            except ValueError:
                max_len = 256
            outputs = model.generate(input_ids, max_length=max_len)
            st.session_state["generated_text"] = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.session_state["processed_input"] = st.session_state["main_input"]
            if st.session_state["config"]["DIFFS"][str(st.session_state["action"]).lower()].strip() == "True":
                annotated_text(*self.annotate(st.session_state["main_input"], st.session_state["generated_text"]))
            else:
                st.markdown(st.session_state["generated_text"])

    def run(self):       
        st.markdown("""<style> .block-container {
                        padding-top:    0.8rem;
                        padding-bottom: 0rem;
                        padding-left:   5rem;
                        padding-right:  5rem;
                    } </style>""", unsafe_allow_html=True)
                
        st.title("Checkr")

        action = st.segmented_control(
            "Select Action",
            options=[a.title() for a in st.session_state["config"]["PROMPTS"].keys()],
            key="LLM_action",
            default=[a.title() for a in st.session_state["config"]["PROMPTS"].keys()][0],
        )
        st.session_state["action"] = action

        with st.sidebar:
            st.sidebar.title("Configurations")
            if "ollama_settings_prompts_dialog" not in st.session_state:
                if st.button("⚙️ Settings", use_container_width=True):
                    self.model_prompt_setup()

            if st.button("🔄 Restore Defaults", use_container_width=True):
                self.load_config(defaults=True)


        col_input, col_output = st.columns(2)
        with col_input:
            input_text = st.text_area(
                "Input Text",
                label_visibility="hidden",
                value="",
                height=500,
                key="main_input"
            )
        with col_output:
            st.write('<div style="height: 44px;"></div>', unsafe_allow_html=True)
            if (input_text is not None) and (input_text.strip() != "") and \
                "processed_input" in st.session_state and \
                (input_text != str(st.session_state.get(f"processed_input", ""))):
                self.run_llm()

        st.session_state["processed_input"] = input_text
        self.footer()

    def footer(self):
        st.markdown(
            f"""
            <style>
            .footer {{
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: #0e1117;
                color: gray;
                text-align: center;
                padding: 10px 0;
            }}
            </style>
            
            <div class="footer">
                Developed with 🩶 by Meinhard Kissich
            </div>
            """,
            unsafe_allow_html=True,
        )



if __name__ == "__main__":
    app = App()
    app.run()

