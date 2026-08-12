"""
Layer 2: Project Graph DB (Neo4j)
Builds a queryable graph on top of the relational (Postgres/SQLite) data so
Layer 3 can traverse relationships directly (e.g. "show every material this
vendor supplies across all projects and their current risk").

Node types: Project, Vendor, Material, PurchaseOrder, Submittal,
            VendorComm, Shipment, ScheduleTask
Relationships:
  (Project)-[:CONTAINS]->(Material)
  (Vendor)-[:SUPPLIES]->(Material)
  (Material)-[:ORDERED_VIA]->(PurchaseOrder)-[:PLACED_WITH]->(Vendor)
  (Material)-[:HAS_SUBMITTAL]->(Submittal)
  (Material)-[:HAS_SHIPMENT]->(Shipment)
  (Vendor)-[:SENT]->(VendorComm)-[:REGARDING]->(Material)
  (Material)-[:REQUIRED_BY]->(ScheduleTask)
"""

import os
try:
    from neo4j import GraphDatabase
except Exception:
    GraphDatabase = None

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")


class GraphDBClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) if GraphDatabase else None

    def close(self):
        if self.driver:
            self.driver.close()

    def _run(self, query: str, **params):
        if not self.driver:
            raise RuntimeError("Neo4j driver is unavailable; install neo4j or run without graph sync")
        with self.driver.session() as session:
            session.run(query, **params)

    # -------------------- Node upserts --------------------

    def upsert_project(self, project_id: str, project_name: str, location: str = None):
        self._run(
            """
            MERGE (p:Project {project_id: $project_id})
            SET p.name = $project_name, p.location = $location
            """,
            project_id=project_id, project_name=project_name, location=location,
        )

    def upsert_vendor(self, vendor_id: str, vendor_name: str, reliability_score: float = None):
        self._run(
            """
            MERGE (v:Vendor {vendor_id: $vendor_id})
            SET v.name = $vendor_name, v.reliability_score = $reliability_score
            """,
            vendor_id=vendor_id, vendor_name=vendor_name, reliability_score=reliability_score,
        )

    def upsert_material(self, material_id: str, project_id: str, description: str,
                         sku: str = None, material_class: str = None):
        self._run(
            """
            MERGE (m:Material {material_id: $material_id})
            SET m.description = $description, m.sku = $sku, m.material_class = $material_class
            WITH m
            MATCH (p:Project {project_id: $project_id})
            MERGE (p)-[:CONTAINS]->(m)
            """,
            material_id=material_id, project_id=project_id,
            description=description, sku=sku, material_class=material_class,
        )

    def upsert_purchase_order(self, po_id: str, material_id: str, vendor_id: str,
                               po_number: str, order_date, promised_ship_date):
        self._run(
            """
            MERGE (po:PurchaseOrder {po_id: $po_id})
            SET po.po_number = $po_number,
                po.order_date = toString($order_date),
                po.promised_ship_date = toString($promised_ship_date)
            WITH po
            MATCH (m:Material {material_id: $material_id})
            MERGE (m)-[:ORDERED_VIA]->(po)
            WITH po, m
            MATCH (v:Vendor {vendor_id: $vendor_id})
            MERGE (po)-[:PLACED_WITH]->(v)
            MERGE (v)-[:SUPPLIES]->(m)
            """,
            po_id=po_id, material_id=material_id, vendor_id=vendor_id,
            po_number=po_number, order_date=order_date, promised_ship_date=promised_ship_date,
        )

    def upsert_submittal(self, submittal_id: str, material_id: str,
                          approval_status: str, submitted_date):
        self._run(
            """
            MERGE (s:Submittal {submittal_id: $submittal_id})
            SET s.approval_status = $approval_status, s.submitted_date = toString($submitted_date)
            WITH s
            MATCH (m:Material {material_id: $material_id})
            MERGE (m)-[:HAS_SUBMITTAL]->(s)
            """,
            submittal_id=submittal_id, material_id=material_id,
            approval_status=approval_status, submitted_date=submitted_date,
        )

    def upsert_shipment(self, shipment_id: str, material_id: str, status: str,
                         shipped_date, estimated_arrival, lead_time_days: float = None,
                         delay_bucket: str = None):
        self._run(
            """
            MERGE (sh:Shipment {shipment_id: $shipment_id})
            SET sh.status = $status,
                sh.shipped_date = toString($shipped_date),
                sh.estimated_arrival = toString($estimated_arrival),
                sh.lead_time_days = $lead_time_days,
                sh.delay_bucket = $delay_bucket
            WITH sh
            MATCH (m:Material {material_id: $material_id})
            MERGE (m)-[:HAS_SHIPMENT]->(sh)
            """,
            shipment_id=shipment_id, material_id=material_id, status=status,
            shipped_date=shipped_date, estimated_arrival=estimated_arrival,
            lead_time_days=lead_time_days, delay_bucket=delay_bucket,
        )

    def upsert_schedule_task(self, schedule_id: str, material_id: str, task_name: str,
                              roj_date, is_critical_path: bool):
        self._run(
            """
            MERGE (t:ScheduleTask {schedule_id: $schedule_id})
            SET t.task_name = $task_name,
                t.roj_date = toString($roj_date),
                t.is_critical_path = $is_critical_path
            WITH t
            MATCH (m:Material {material_id: $material_id})
            MERGE (m)-[:REQUIRED_BY]->(t)
            """,
            schedule_id=schedule_id, material_id=material_id, task_name=task_name,
            roj_date=roj_date, is_critical_path=is_critical_path,
        )

    def upsert_vendor_comm(self, comm_id: str, vendor_id: str, material_id: str,
                            comm_type: str, message_date, extracted_summary: str, delay_days_mentioned: int = None):
        self._run(
            """
            MERGE (c:VendorComm {comm_id: $comm_id})
            SET c.comm_type = $comm_type,
                c.message_date = toString($message_date),
                c.summary = $extracted_summary,
                c.delay_days_mentioned = $delay_days_mentioned
            WITH c
            MATCH (v:Vendor {vendor_id: $vendor_id})
            MERGE (v)-[:SENT]->(c)
            WITH c
            MATCH (m:Material {material_id: $material_id})
            MERGE (c)-[:REGARDING]->(m)
            """,
            comm_id=comm_id, vendor_id=vendor_id, material_id=material_id,
            comm_type=comm_type, message_date=message_date, extracted_summary=extracted_summary,
            delay_days_mentioned=delay_days_mentioned,
        )

    # -------------------- Query helpers for Layer 3 --------------------

    def get_material_full_context(self, material_id: str) -> list:
        """Returns everything linked to a material: vendor, POs, shipments, schedule, comms."""
        if not self.driver:
            return []
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (m:Material {material_id: $material_id})
                OPTIONAL MATCH (m)-[:ORDERED_VIA]->(po:PurchaseOrder)-[:PLACED_WITH]->(v:Vendor)
                OPTIONAL MATCH (m)-[:HAS_SHIPMENT]->(sh:Shipment)
                OPTIONAL MATCH (m)-[:REQUIRED_BY]->(t:ScheduleTask)
                OPTIONAL MATCH (c:VendorComm)-[:REGARDING]->(m)
                RETURN m, po, v, sh, t, c
                """,
                material_id=material_id,
            )
            return [record.data() for record in result]


# Singleton instance to import elsewhere (e.g. main_layer1.py, feature_api_layer2.py)
graph_client = GraphDBClient()
