"""
Tests for POST /deployments/impact endpoint (orchestrator dry-run).
"""

import asyncio
from unittest import mock

import pytest

from datajunction_server.models.deployment import (
    ColumnSpec,
    DeploymentInfo,
    DeploymentSpec,
    SourceSpec,
    TransformSpec,
)


@pytest.fixture(autouse=True, scope="module")
def patch_effective_writer_concurrency():
    from datajunction_server.internal.deployment.deployment import settings

    with mock.patch.object(
        settings.__class__,
        "effective_writer_concurrency",
        new_callable=mock.PropertyMock,
        return_value=1,
    ):
        yield


async def _wait_for_deployment(client, deployment_id: str, timeout: int = 30):
    """Poll until a deployment reaches a terminal status."""
    for _ in range(timeout):
        resp = await client.get(f"/deployments/{deployment_id}")
        if resp.json()["status"] in ("success", "failed"):
            return resp.json()
        await asyncio.sleep(0.1)
    return (await client.get(f"/deployments/{deployment_id}")).json()


async def _impact_nodes(client, spec):
    response = await client.post(
        "/deployments/impact",
        json=spec.model_dump(by_alias=True),
    )
    assert response.status_code == 200
    return {
        result["name"]: result
        for result in response.json()["results"]
        if result["deploy_type"] == "node"
    }


class TestDeploymentImpactEndpoint:
    """Tests for POST /deployments/impact (orchestrator dry-run)."""

    @pytest.mark.asyncio
    async def test_impact_create_new_nodes(self, client_with_roads):
        """Deploying into an empty namespace: results show CREATE operations."""
        spec = DeploymentSpec(
            namespace="impact_create_test",
            nodes=[
                SourceSpec(
                    name="orders",
                    catalog="default",
                    schema_="test",
                    table="orders",
                    columns=[
                        ColumnSpec(name="order_id", type="int"),
                        ColumnSpec(name="amount", type="float"),
                    ],
                ),
            ],
        )

        response = await client_with_roads.post(
            "/deployments/impact",
            json=spec.model_dump(by_alias=True),
        )
        assert response.status_code == 200

        data = response.json()
        info = DeploymentInfo(**data)
        assert info.uuid == "dry_run"
        assert info.namespace == "impact_create_test"

        node_results = [r for r in info.results if r.deploy_type == "node"]
        assert len(node_results) == 1
        assert node_results[0].name == "impact_create_test.orders"
        assert node_results[0].operation == "create"
        assert node_results[0].change_tier == "major"
        assert (
            node_results[0].semantic_fingerprint == spec.nodes[0].semantic_fingerprint()
        )

    @pytest.mark.asyncio
    async def test_impact_detects_updates(self, client_with_roads):
        """After deploying a node, a dry-run with a changed query shows UPDATE."""
        initial_spec = DeploymentSpec(
            namespace="impact_update_test",
            nodes=[
                TransformSpec(
                    name="orders_summary",
                    query="SELECT 1 AS order_id FROM ${prefix}raw",
                ),
                SourceSpec(
                    name="raw",
                    catalog="default",
                    schema_="test",
                    table="raw",
                    columns=[ColumnSpec(name="order_id", type="int")],
                ),
            ],
        )

        deploy_resp = await client_with_roads.post(
            "/deployments",
            json=initial_spec.model_dump(by_alias=True),
        )
        assert deploy_resp.status_code == 200
        await _wait_for_deployment(client_with_roads, deploy_resp.json()["uuid"])

        updated_spec = DeploymentSpec(
            namespace="impact_update_test",
            nodes=[
                TransformSpec(
                    name="orders_summary",
                    query="SELECT 1 AS order_id, 'updated' AS status FROM ${prefix}raw",
                ),
                SourceSpec(
                    name="raw",
                    catalog="default",
                    schema_="test",
                    table="raw",
                    columns=[ColumnSpec(name="order_id", type="int")],
                ),
            ],
        )

        response = await client_with_roads.post(
            "/deployments/impact",
            json=updated_spec.model_dump(by_alias=True),
        )
        assert response.status_code == 200

        data = response.json()
        update_results = [
            r
            for r in data["results"]
            if r["operation"] == "update" and r["deploy_type"] == "node"
        ]
        skip_results = [r for r in data["results"] if r["operation"] == "noop"]
        assert len(update_results) == 1
        assert "orders_summary" in update_results[0]["name"]
        assert len(skip_results) >= 1
        assert update_results[0]["change_tier"] == "major"
        assert (
            update_results[0]["semantic_fingerprint"]
            == updated_spec.nodes[0].semantic_fingerprint().model_dump()
        )
        assert (
            update_results[0]["semantic_fingerprint"]
            != initial_spec.nodes[0].semantic_fingerprint().model_dump()
        )
        unchanged_nodes = [
            result for result in skip_results if result["deploy_type"] == "node"
        ]
        assert all(result["change_tier"] == "none" for result in unchanged_nodes)
        assert all(result["semantic_fingerprint"] for result in unchanged_nodes)

    @pytest.mark.asyncio
    async def test_impact_detects_deletions(self, client_with_roads):
        """Nodes present in DB but absent from spec appear as DELETE in dry-run."""
        initial_spec = DeploymentSpec(
            namespace="impact_delete_test",
            nodes=[
                SourceSpec(
                    name="to_keep",
                    catalog="default",
                    schema_="test",
                    table="keep",
                    columns=[ColumnSpec(name="id", type="int")],
                ),
                SourceSpec(
                    name="to_delete",
                    catalog="default",
                    schema_="test",
                    table="delete_me",
                    columns=[ColumnSpec(name="id", type="int")],
                ),
            ],
        )

        deploy_resp = await client_with_roads.post(
            "/deployments",
            json=initial_spec.model_dump(by_alias=True),
        )
        assert deploy_resp.status_code == 200
        await _wait_for_deployment(client_with_roads, deploy_resp.json()["uuid"])

        modified_spec = DeploymentSpec(
            namespace="impact_delete_test",
            nodes=[
                SourceSpec(
                    name="to_keep",
                    catalog="default",
                    schema_="test",
                    table="keep",
                    columns=[ColumnSpec(name="id", type="int")],
                ),
                # to_delete omitted
            ],
        )

        response = await client_with_roads.post(
            "/deployments/impact",
            json=modified_spec.model_dump(by_alias=True),
        )
        assert response.status_code == 200

        data = response.json()
        delete_results = [r for r in data["results"] if r["operation"] == "delete"]
        assert len(delete_results) == 1
        assert "to_delete" in delete_results[0]["name"]
        assert delete_results[0]["change_tier"] == "major"
        assert (
            delete_results[0]["semantic_fingerprint"]
            == initial_spec.nodes[1].semantic_fingerprint().model_dump()
        )

    @pytest.mark.asyncio
    async def test_impact_minor_full_noop_and_forced_revalidation(
        self,
        client_with_roads,
    ):
        initial = DeploymentSpec(
            namespace="impact_tiers_test",
            nodes=[
                SourceSpec(
                    name="one",
                    catalog="default",
                    schema_="test",
                    table="one",
                    description="Before",
                    columns=[ColumnSpec(name="id", type="int")],
                ),
                SourceSpec(
                    name="two",
                    catalog="default",
                    schema_="test",
                    table="two",
                    columns=[ColumnSpec(name="id", type="int")],
                ),
            ],
        )
        deployed = await client_with_roads.post(
            "/deployments",
            json=initial.model_dump(by_alias=True),
        )
        await _wait_for_deployment(client_with_roads, deployed.json()["uuid"])

        minor = initial.model_copy(deep=True)
        minor.nodes[0].description = "After"
        by_name = await _impact_nodes(client_with_roads, minor)
        assert by_name["impact_tiers_test.one"]["change_tier"] == "minor"
        assert (
            by_name["impact_tiers_test.one"]["semantic_fingerprint"]
            == initial.nodes[0].semantic_fingerprint().model_dump()
        )
        assert by_name["impact_tiers_test.two"]["change_tier"] == "none"

        noop_nodes = await _impact_nodes(client_with_roads, initial)
        assert set(noop_nodes) == {
            "impact_tiers_test.one",
            "impact_tiers_test.two",
        }
        assert all(result["change_tier"] == "none" for result in noop_nodes.values())
        assert all(result["semantic_fingerprint"] for result in noop_nodes.values())

        forced = initial.model_copy(update={"force": True})
        forced_nodes = await _impact_nodes(client_with_roads, forced)
        assert all(result["operation"] == "update" for result in forced_nodes.values())
        assert all(result["change_tier"] == "none" for result in forced_nodes.values())
        assert all(result["semantic_fingerprint"] for result in forced_nodes.values())

    @pytest.mark.asyncio
    async def test_dry_run_does_not_mutate_db(self, client_with_roads):
        """Calling /deployments/impact must not persist any changes."""
        spec = DeploymentSpec(
            namespace="dry_run_no_mutation_test",
            nodes=[
                SourceSpec(
                    name="ephemeral",
                    catalog="default",
                    schema_="test",
                    table="ephemeral",
                    columns=[ColumnSpec(name="id", type="int")],
                ),
            ],
        )

        # Call dry-run
        impact_resp = await client_with_roads.post(
            "/deployments/impact",
            json=spec.model_dump(by_alias=True),
        )
        assert impact_resp.status_code == 200
        assert impact_resp.json()["uuid"] == "dry_run"

        # The node must NOT exist in the namespace
        nodes_resp = await client_with_roads.get(
            "/namespaces/dry_run_no_mutation_test/nodes/",
        )
        assert nodes_resp.status_code in (200, 404)
        if nodes_resp.status_code == 200:
            node_names = [n["name"] for n in nodes_resp.json()]
            assert "dry_run_no_mutation_test.ephemeral" not in node_names

    @pytest.mark.asyncio
    async def test_invalid_fingerprint_fails_preview(self, client_with_roads):
        spec = DeploymentSpec(
            namespace="impact_bad_fingerprint",
            nodes=[
                SourceSpec(
                    name="source",
                    catalog="default",
                    schema_="test",
                    table="source",
                ),
            ],
        )
        with mock.patch.object(
            SourceSpec,
            "semantic_fingerprint",
            side_effect=ValueError("cannot fingerprint"),
        ):
            response = await client_with_roads.post(
                "/deployments/impact",
                json=spec.model_dump(by_alias=True),
            )
        assert response.status_code == 500
        assert response.json() == {"detail": "Internal Server Error"}

    @pytest.mark.asyncio
    async def test_downstream_impacts_returned_for_invalid_parent(
        self,
        client_with_roads,
    ):
        """If a parent node is changed to INVALID, downstream nodes appear in
        downstream_impacts with impact_type=will_invalidate."""
        # Deploy a transform that depends on a source
        initial_spec = DeploymentSpec(
            namespace="impact_downstream_test",
            nodes=[
                SourceSpec(
                    name="base",
                    catalog="default",
                    schema_="test",
                    table="base",
                    columns=[
                        ColumnSpec(name="id", type="int"),
                        ColumnSpec(name="value", type="float"),
                    ],
                ),
                TransformSpec(
                    name="derived",
                    query="SELECT id, value FROM ${prefix}base",
                    owners=["dj"],
                ),
            ],
        )

        deploy_resp = await client_with_roads.post(
            "/deployments",
            json=initial_spec.model_dump(by_alias=True),
        )
        assert deploy_resp.status_code == 200
        await _wait_for_deployment(client_with_roads, deploy_resp.json()["uuid"])

        # Dry-run: remove the 'value' column from base, making the transform invalid
        modified_spec = DeploymentSpec(
            namespace="impact_downstream_test",
            nodes=[
                SourceSpec(
                    name="base",
                    catalog="default",
                    schema_="test",
                    table="base",
                    columns=[
                        ColumnSpec(name="id", type="int"),
                        # 'value' removed
                    ],
                ),
                TransformSpec(
                    name="derived",
                    query="SELECT id, value FROM ${prefix}base",  # will break
                    owners=["dj"],
                ),
            ],
        )

        response = await client_with_roads.post(
            "/deployments/impact",
            json=modified_spec.model_dump(by_alias=True),
        )
        assert response.status_code == 200

        data = response.json()
        # At minimum the request succeeds and returns the expected shape
        assert "results" in data
        assert "downstream_impacts" in data
        assert isinstance(data["downstream_impacts"], list)

        # Every impact carries the owners of the affected node
        derived = [
            impact
            for impact in data["downstream_impacts"]
            if impact["name"] == "impact_downstream_test.derived"
        ]
        assert derived == [
            {
                "name": "impact_downstream_test.derived",
                "node_type": "transform",
                "current_status": "valid",
                "predicted_status": "invalid",
                "impact_type": "will_invalidate",
                "impact_reason": (
                    "Revalidation failed: Column `value` not found in any table. "
                    "Available tables: ['impact_downstream_test.base']"
                ),
                "depth": 1,
                "caused_by": ["impact_downstream_test.base"],
                "is_external": False,
                "owners": ["dj"],
            },
        ]
