import os
from apikey import apikey
from pdfloader import PDF_Loader
from langchain.llms import OpenAI
import streamlit as st

def main():
    os.environ['OPENAI_API_KEY'] = apikey
    
    link = "https://watermark.silverchair.com/zdc10111000s62.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAsQwggLABgkqhkiG9w0BBwagggKxMIICrQIBADCCAqYGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQM0QZULkXYp4HTYlAnAgEQgIICd_gL77IJyq3VMQNWw33lXqnmKEDNfBGr_49CT1N58sl1r5rX1Ga3thKvrCxwocvNt9791ayEXbxIXLip6XkhE92rLmGtW70ek6hNji1zPkz-ThUI2Az2tMYJTYT_gGEYC5exJr1hmtNcpmNvYO2-Vn0RW1rOts3ciw22VKAJyTgZkxLH0R0kHudtC27S23khx0iYzMWLT4ZoaARvyw0CjH5t3648S9O1UpyfqXoon7TR2PtvjKq7nhC8qJUHnzvVKeOAWjjy9W4XEE45yLz1pS-sMiLuL7X6Dxb2RSEdSt1_X5EGqnt-Y06QduLkgWgPqTrG_3KVEpVHRXBaxhEFUCdfLRa6C_l7kfQAuePf7oixL-1eelzXyH_OONi8W5jVOw78WmpqLVAZp0APghFErROOzPVOpBXFjyfhRwSvSaq--NCR4CaObM6qGXicrLe2aCqWL0cICybmFPoo4ENIxTz5udF2hrt0CGQ2ASxJPV4Qd37bVNFm2_5ITi3faErDQF8v6NoOMxehjAiSFsCA5MD_J-Ubia0YDnRoMuuo-0NzvTrbeiMpJEKzCnqoqym6m4mu3rawVWz3DiKWSgiR9ZAEpW3JdMxNCJdmetTzQH94m68vjDkxm-TCv1VD_ZrE8FWhczn9N76bCXj51WM84_Xw54KNniZ-rV3wTfp_MuxY6xKLZiAKBAscEgJ_jbPaKVe17UqGkN1O_c001QLs96ojh0BwXuARM57NBh8YTiBo1SJBGtiktQfPEEFj4Mo15hx0BVtD0tE3B1PM6AwS1MVsIyogBYtUDB3O6To0RmKhcmW_kJddF0Zi7ghY5kck9blAwKvcw_w"

    # Display the page title and the text box for the user to ask the question
    st.title('🦜 Search and query academic medical papers ')

    #Take in some prompt from the user
    # TODO: Create some context for the bot so that it knows what it's purpose is
    # -Add additional agents and tools that can be accessed beyond the chatgpt API training
    # -Find some way to navigate to the PDF version of the database/paper, then feed that url to the pdf loader
    prompt = st.text_input("What medical topic would you like to know about?")

    vectorized_doc = PDF_Loader.loadPDF(url=link)

    PDF_Loader.queryPDF(prompt, vectorized_doc)
    
    llm = OpenAI(temperature = 0.9)

    if prompt:

        response = llm(prompt)
        st.write(response)


if __name__ == '__main__':
    main()




