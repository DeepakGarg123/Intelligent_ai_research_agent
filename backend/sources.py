def extract_document_sources(results):
    sources = []
    seen_sources = set()
    
    for document , score in results:
        metadata = document.metadata
        
        source = metadata.get("source" , "Unknown")
        page = metadata.get("page_label" , metadata.get("page" , "Unknown"))
        
        source_info = (source , page)
        
        if source_info not in seen_sources:
            seen_sources.add(source_info)
            
            sources.append({
                "type":"document",
                "file":source , 
                "page":page
            })
            
        return sources
        