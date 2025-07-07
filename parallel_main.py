import os
import time
import argparse
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from sqlalchemy import create_engine
from schema_engine import SchemaEngine
from parallel_schema_engine import ParallelSchemaEngine


def main():
    parser = argparse.ArgumentParser(description="Process database schema with parallel or sequential execution")
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    parser.add_argument('--db_path', type=str, default='./book_1.sqlite', help='Path to the database file')
    parser.add_argument('--output_json', type=str, default='./output_schema.json', help='Output JSON file path')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Maximum number of worker threads for parallel processing')
    parser.add_argument('--comment_mode', type=str, default='generation',
                        choices=['origin', 'merge', 'generation', 'no_comment'],
                        help='Comment mode for schema generation')
    parser.add_argument('--api_key', type=str, default='YOUR API KEY HERE.', help='DashScope API key')
    args = parser.parse_args()

    # Initialize LLM
    dashscope_llm = DashScope(model_name=DashScopeGenerationModels.QWEN_PLUS, api_key=args.api_key)

    # Get absolute path to database
    db_abs_path = os.path.abspath(args.db_path)
    db_engine = create_engine(f'sqlite:///{db_abs_path}')

    # Get database name from the file path
    db_name = os.path.splitext(os.path.basename(args.db_path))[0]

    # Start timing
    start_time = time.time()

    # Choose between parallel and sequential processing
    if args.parallel:
        print(f"Using parallel processing with {args.max_workers} workers")
        schema_engine_instance = ParallelSchemaEngine(
            db_engine,
            llm=dashscope_llm,
            db_name=db_name,
            comment_mode=args.comment_mode,
            max_workers=args.max_workers
        )
    else:
        print("Using sequential processing")
        schema_engine_instance = SchemaEngine(
            db_engine,
            llm=dashscope_llm,
            db_name=db_name,
            comment_mode=args.comment_mode
        )

    # Process the database schema
    print("Categorizing fields...")
    schema_engine_instance.fields_category()

    print("Generating table and column descriptions...")
    schema_engine_instance.table_and_column_desc_generation()

    # Save the result
    mschema = schema_engine_instance.mschema
    mschema.save(args.output_json)

    # Print the result
    mschema_str = mschema.to_mschema()
    print(mschema_str)

    # Print execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()