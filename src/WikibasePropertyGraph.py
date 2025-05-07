import json
import os
from wikibaseintegrator import WikibaseIntegrator
from wikibaseintegrator import wbi_helpers
from wikibaseintegrator import wbi_login, WikibaseIntegrator
from wikibaseintegrator.wbi_config import config as wbi_config
from typing import Tuple, Any, List, Optional, Dict

from llama_index.core.graph_stores import (
    SimplePropertyGraphStore,
    EntityNode,
    ChunkNode
)
from llama_index.core.graph_stores.types import LabelledNode, VectorStoreQuery, VECTOR_SOURCE_KEY, KG_SOURCE_REL, Relation, LabelledPropertyGraph
from llama_index.core.schema import TextNode

class WikibasePropertyGraphStore(SimplePropertyGraphStore):
    def __init__(
        self,
        embed_model = Any,
        vector_store = Any, #BasePydanticVectorStore
        graph: Optional[LabelledPropertyGraph] = None
    ) -> None:
        SimplePropertyGraphStore.__init__(self) #call parent class init
        self.wb_items = {}
        self.wb_properties = {}
        #print(wb_api_url)
        #print(username)
        #print(pw)
        #print(self.wb_items)
        #print(self.wb_properties)
        
        self.language = 'en'
        self.embed_model = embed_model
        self.vector_store = vector_store
        self.supports_vector_queries: bool = True
        if graph:
            self.graph = graph

    def login(self,url, username, pw):
        wbi_config['MEDIAWIKI_API_URL'] = url
        wbi_config['SPARQL_ENDPOINT_URL'] = url+':8834/proxy/wdqs/bigdata/namespace/wdq/sparql'
        wbi_config['WIKIBASE_URL'] = 'http://wikibase.svc'
        login_instance = wbi_login.Clientlogin(user=username, password=pw)
        wbi_config['USER_AGENT'] = 'MyWikibaseBot/1.0'
        wbi = WikibaseIntegrator(login=login_instance)
        return wbi
    
    def init_graph_from_wiki(
            self,
            wb_api_url: str,
            username = str,
            pw = str
            ) -> None:
        self.wbapi_url = wb_api_url
        self.wbi = self.login(wb_api_url, username, pw)
        self.fetchWikibaseContent()
        
        graph_nodes = []
        graph_relations = []
        for item_id in self.wb_items:
            """ add wikibase items to graph"""
            wbitem = self.wb_items[item_id]
            
            node_name = self.getWikibaseItemName(wbitem)
            if self.language in wbitem['labels']:
                node_label = wbitem['labels'][self.language]['value']   

            node: EntityNode = EntityNode(name=node_name,label=node_label) 

            if self.language in wbitem['descriptions']:
                node.properties['wbdescription'] = wbitem['descriptions'][self.language]['value']  
            
            embeddingtext = "label: "+ node.label
            if 'wbdescription' in node.properties:
                embeddingtext = embeddingtext + " description: "+node.properties['wbdescription']
            embedding = self.embed_model.get_text_embedding( embeddingtext )
            node.embedding = embedding
            graph_nodes.append(node)
            if item_id == "Q4950":
                print(wbitem['claims'])
            """ process wikibase item statements """  
            for pid in wbitem['claims']:
                for claim_dict in wbitem['claims'][pid]:
                    if claim_dict['type'] == 'statement':
                        if claim_dict['mainsnak']['datatype'] == 'wikibase-item':
                            """ insert item-to-item statements as relationships """
                            #get property label from previous fetched data
                            property = self.wb_properties[claim_dict['mainsnak']['property']]
                            if self.language in property['labels']:
                                label = property['labels'][self.language]['value']
                            else:
                                label = property['id']
                            relation_properties = {}
                            if self.language in property['descriptions']:
                                relation_properties['wbdescription'] = property['descriptions'][self.language]['value']
                            if 'datavalue' in claim_dict['mainsnak']:
                                target_wbid = claim_dict['mainsnak']['datavalue']['value']['id']
                            else:
                                print("item "+node_name+": no mainsnak->datavalue in claim:"+str(claim_dict)+" skipped claim!")
                                continue
                            if target_wbid in self.wb_items:
                                target_name = self.getWikibaseItemName(self.wb_items[target_wbid])
                            else:
                                print(str(target_wbid)+" is not in wbitems")
                                continue
                            relation = Relation(
                                label = label,
                                source_id = node_name,
                                target_id = target_name,
                                properties = relation_properties #TODO: add claim qualifier as properties
                            )
                            graph_relations.append(relation)
                            #add relation also as property as llama_index is not using the propery part of triples in VectorcontextRetriever( 7.5.2025)
                            label = self.getWikibaseItemName(self.wb_properties[claim_dict['mainsnak']['property']])
                            value = self.getWikibaseItemName(self.wb_items[claim_dict['mainsnak']['datavalue']['value']['id']])
                            if label not in node.properties:
                                node.properties[label] = []
                            node.properties[label].append(value) # a statement can occur multiple times with different values
                        
                        else:
                            """ insert value statements as properties """
                            # values cannot be displayed as relations as the value has is no unique entry in the db, featuring a target id
                            label = self.getWikibaseItemName(self.wb_properties[claim_dict['mainsnak']['property']])
                            value = claim_dict['mainsnak']['datavalue']['value']
                            if label not in node.properties:
                                node.properties[label] = []
                            node.properties[label].append(value) # a statement can occur multiple times with different values


        self.upsert_nodes(graph_nodes)
        print("init vector store...")
        self._insert_nodes_to_vector_index(graph_nodes)
        self.upsert_relations(graph_relations)
        print("graph initialized")
        #print(self.get())
    
    def getWikibaseItemName(self, wbitem):
        node_name = wbitem['id']        
        if self.language in wbitem['labels']:
            node_name = node_name + ": " + wbitem['labels'][self.language]['value']
        return node_name
        

    def _insert_nodes_to_vector_index(self, nodes: List[LabelledNode]) -> None:
        """Insert vector nodes."""
        #print(self.vector_store)
        assert self.vector_store is not None
      
        llama_nodes: List[TextNode] = []
        for node in nodes:
            if node.embedding is not None:
                llama_nodes.append(
                    TextNode(
                        text=str(node),
                        metadata={VECTOR_SOURCE_KEY: node.id, **node.properties},
                        embedding=[*node.embedding],
                    )
                )
                if not self.vector_store.stores_text:
                    llama_nodes[-1].id_ = node.id

            # clear the embedding to save memory, its not used now
            node.embedding = None
        #print(llama_nodes)
        self.vector_store.add(llama_nodes)

    def vector_query(
            self, query: VectorStoreQuery, **kwargs: Any
        ) -> Tuple[List[LabelledNode], List[float]]:
        """Query the vector store with a vector store query."""
        #print("query: "+str(query))
        vectorstore_result = self.vector_store.query(query, **kwargs)
        print("vectorstore_result:"+str(vectorstore_result))
        graph_nodes = self.get(ids=[n.metadata['vector_source_id'] for n in vectorstore_result.nodes])
        #all_graph_nodes = self.get()
        #graph_nodes = [all_graph_nodes[int(i)] for i in vectorstore_result.ids]
        #graph_nodes = [all_graph_nodes[int(i)] for i in vectorstore_result.ids]
        result = [graph_nodes, vectorstore_result.similarities]
        
        #print(result)
        #kg_ids = [node.id for node in graph_nodes]
        #triplets = self.get_rel_map(
        #        graph_nodes,
        #        depth=1,
        #        limit=2,
        #        ignore_rels=[KG_SOURCE_REL],
        #    )
        #print(triplets)
        return result
    
    def getAllWikibasePageIds(self, namespace, limit=10, continueParam_name=None, continueParam_value=None) -> List[str]:
        wbids = []
        continue_params = {}

        while True:
            apicall = {
                "action": "query",
                "list": "allpages",
                "apnamespace": namespace,
                "format": "json",
                "formatversion": "2",
                "aplimit": limit
            }

            # If there are continuation parameters, include them
            apicall.update(continue_params)

            result_obj = wbi_helpers.mediawiki_api_call(
                method="POST",
                mediawiki_api_url=self.wbi.login.mediawiki_api_url,
                data=apicall
            )

            for page in result_obj.get('query', {}).get('allpages', []):
                wbids.append(page['title'].replace("Item:", "").replace("Property:", ""))

            # Handle pagination
            if "continue" in result_obj:
                continue_params = result_obj['continue']
            else:
                break

        print(f"{len(wbids)} Wikibase items found in namespace: "+str(namespace))
        return wbids
    
    def fetchWikibaseContent(self):
        wb_item_ids = self.getAllWikibasePageIds(120, 500)
        wb_prop_ids = self.getAllWikibasePageIds(122, 500)
        all_ids = wb_item_ids + wb_prop_ids
        #print(all_ids)
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        sep = "|"
        for id_chunk in chunks(all_ids, 10):
            apicall = {
                "action": "wbgetentities",
                "ids": sep.join(id_chunk),
                "format": "json",
                "formatversion": "2",
            }
            result_obj = wbi_helpers.mediawiki_api_call(
                method="POST",
                mediawiki_api_url=self.wbi.login.mediawiki_api_url,
                data=apicall
            )
            #if "Q5457" in id_chunk:
            #    print(result_obj)

            for obj_id in result_obj.get('entities', {}):
                obj = result_obj['entities'][obj_id]
                if obj['type'] == "item":
                    if 'redirects' in obj: #if this is a redirect, create a copy
                        id = obj['redirects']['from']
                    else:
                        id = obj['id']
                    self.wb_items[id] = obj
                elif obj['type'] == 'property':
                    self.wb_properties[obj['id']] = obj   
        
    @classmethod
    def from_dict(
        cls,
        data: dict,
        embed_model = Any,
        vector_store = Any #BasePydanticVectorStore
    ) -> "WikibasePropertyGraphStore":
        """Load from dict."""
        # need to load nodes manually
        node_dicts = data["nodes"]

        kg_nodes: Dict[str, LabelledNode] = {}
        for id, node_dict in node_dicts.items():
            if "name" in node_dict:
                kg_nodes[id] = EntityNode.model_validate(node_dict)
            elif "text" in node_dict:
                kg_nodes[id] = ChunkNode.model_validate(node_dict)
            else:
                raise ValueError(f"Could not infer node type for data: {node_dict!s}")

        # clear the nodes, to load later
        data["nodes"] = {}

        # load the graph
        graph = LabelledPropertyGraph.model_validate(data)

        # add the node back
        graph.nodes = kg_nodes

        return cls(graph=graph, embed_model=embed_model, vector_store=vector_store)
    
        
      
        

    
        
    

