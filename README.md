## Installation

1. Install Antigravity
2. In powershell - copy paste the command to install 'uv' in your project.(command will be in uv website)
3. Now in CMD -> uv init
4. uv venv -> Create a virtual env
5. It will show you to activate the venv with a particular command. Just copy and run it. 
6. Your venv will be activated in cmd.
7. Create requirements.txt for installing your libraries. -> uv add -r requirements.txt -> to install all libraries.
8. Also add ipykernel -> for running your jupyter notebooks.
9. To add any library externally -> uv add 'ipykernel'


## Get API Keys
1. Go to Google AI studio API Key. 
2. Create API Key.
3. Go to Groq -> Create API key.
4. Create .env file in your project and initialize all the API keys there. 


## 1-langchain_intro.ipynb -> Function of AGENTS
1. Create agent and output the agent. You can see a chart -> start -> model -> end
2. Then you have to create tools. Tools is the third party websites/databases that your model will have the access to -> for providing you the data that's exclusively present in the tool.
3. Here as a reference, get_weather() function is used as a tool to retrieve the weather. A generic weather message is being provided here to give a reference. 
4. Then you are using the agent.invoke() to return the message.

## 2-modelintegration.ipynb -> Model Integration, Stream and Batch
1. Integrate different type of models and invoke them.
2. Streaming - Models can stream their output while generating the output. In Invoke, you have to sit still, till you get the full output, while in Streaming, your output will get on generated with a flow. 
3. To use this stream, you have to use model.stream(message) in for loop to get that smooth flow experience of output.
4. Batch - collection of independent requests to a model can reduce costs and improve performance and processing can be done in parallel.
5. use model.batch("multiple messages") and print all the messages in for loop.  
 
## 3-tools.ipynb ->
1. Tools - can perform tasks such as fetching data from database, searching the web or running code. 
2. To make a tool -> specify @tool before the function.
3. To integrate a model with a tool -> model.bind_tools([tool_name])
4. Use invoke to get the contents based on your tool response. 
5. Use the tool execution loops -> to gather the exact response as you have tailored in the tool return value.

## 4-messages.ipynb
1. Messages -> Input and Ouput of model
2. 1 Message contains Role, Content and Metadata.
3. Role -> Human, System, AI message and Tool Message
4. Content -> main content of your output.
5. Metadata -> Optional fields -> use to get this -> message.usage_metadata

## 5-structuredoutput.ipynb -> using Pydantic
1. Models can be requested to provide the response in a format matching given schema.
2. Pydantic provides the richest feature set for field validation, descriptions and nested structures.
3. Create a base class of Movie with all the features -> director, name, ratings, year.-> in your required format.
To create the class of Pydantic use -> class Movie(BaseModel).
4. call the class Movie with model.with_structured_output(Movie).
5. Now if you invoke your normal model by asking about a movie -> it will return all the details about the movie, but you don't want that output, you want the output format that you specified in Movie class.
6. But if you invoke your structured model output model with the same prompt -> it will return the same message but in your required output format.
7. Nested structure -> you can use one class inside your Base class to call mutliple characters. Such as here we use Actor. 
8. In Pydantic, we have validation checks, suppose you have to specify the data type of the variable you want to specify like str for movie name, float for rating, int for year. But, to bypass this checks, we can use Typed Dict.
9. To classify TypedDict , you have to specify it in your class like -> class Movie(TypedDict)
10. We can also use Dataclass, just you have to specify @dataclass before the class.

## 6-middleware.ipynb -> about Middleware
1. Middleware provides a way to tightly control what happens inside an agent. Used for tracking agent behaviour with logging,debugging, Transforming Prompts, tool selection, Adding retries, fallback, Applying rate limits and guardrails. 
2. Basically these are hooks, which are placed before/after the model, tools for some tasks. 
3. Built in Middlewares:
a. Summarization -> provided in the agent -> summarize all the messages within a specified range.
b. Human in the Loop.
c. Model Call Limit
(You can find all the middleware in Langchain docs)
4. Summarization - Based on message length and token length, you can initiate summarization. 