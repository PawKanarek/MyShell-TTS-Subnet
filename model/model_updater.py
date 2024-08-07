import bittensor as bt
import asyncio
from typing import Optional
from constants import CompetitionParameters, COMPETITION_SCHEDULE
import constants
from model.data import ModelMetadata, Model
from model.model_tracker import ModelTracker
from model.storage.local_model_store import LocalModelStore
from model.storage.model_metadata_store import ModelMetadataStore
from model.storage.remote_model_store import RemoteModelStore
from model.utils import get_hash_of_two_strings


class ModelUpdater:
    """Checks if the currently tracked model for a hotkey matches what the miner committed to the chain."""

    def __init__(
        self,
        metadata_store: ModelMetadataStore,
        remote_store: RemoteModelStore,
        local_store: LocalModelStore,
        model_tracker: ModelTracker,
    ):
        self.metadata_store = metadata_store
        self.remote_store = remote_store
        self.local_store = local_store
        self.model_tracker = model_tracker
        self.min_block: Optional[int] = None

    def set_min_block(self, val: Optional[int]):
        self.min_block = val

    @classmethod
    def get_competition_parameters(cls, id: str) -> Optional[CompetitionParameters]:
        for x in COMPETITION_SCHEDULE:
            if x.competition_id == id:
                return x
        return None

    async def _get_metadata(self, hotkey: str) -> Optional[ModelMetadata]:
        """Get metadata about a model by hotkey"""
        return await self.metadata_store.retrieve_model_metadata(hotkey)

    async def sync_models(self, hotkeys: list[str]):
        tasks = [self.sync_model(hotkey) for hotkey in hotkeys]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def sync_model_metadata_only(self, hotkey: str) -> bool:
        """Updates the metadata only for a hotkey if out of sync and returns if it was updated."""
        metadata = await self._get_metadata(hotkey)

        if not metadata:
            bt.logging.trace(
                f"No valid metadata found on the chain for hotkey {hotkey}"
            )
            return False

        if self.min_block and metadata.block < self.min_block:
            bt.logging.trace(
                f"Skipping model for {hotkey} since it was submitted at block {metadata.block} which is less than the minimum block {self.min_block}"
            )
            return False

        # Backwards compatability for models submitted before competition id added
        if metadata.id.competition_id is None:
            metadata.id.competition_id = constants.ORIGINAL_COMPETITION_ID

        parameters = ModelUpdater.get_competition_parameters(metadata.id.competition_id)
        if not parameters:
            bt.logging.trace(
                f"No competition parameters found for {metadata.id.competition_id}"
            )
            return False

        self.model_tracker.on_miner_model_updated_metadata_only(hotkey, metadata)
        return True

    async def ensure_model_downloaded(self, hotkey: str):
        with self.model_tracker.lock:
            if hotkey in self.model_tracker.model_downloaded:
                return

            metadata = self.model_tracker.miner_hotkey_to_model_metadata_dict[hotkey]
            parameters = ModelUpdater.get_competition_parameters(metadata.id.competition_id)

            # Get the local path based on the local store to download to (top level hotkey path)
            path = self.local_store.get_path(hotkey)
            # Otherwise we need to download the new model based on the metadata.
            model = await self.remote_store.download_model(metadata.id, path, parameters)

            # Check that the hash of the downloaded content matches.
            if model.id.hash != metadata.id.hash:
                # If the hash does not match directly, also try it with the hotkey of the miner.
                # This is allowed to help miners prevent same-block copiers.
                hash_with_hotkey = get_hash_of_two_strings(model.id.hash, hotkey)
                if hash_with_hotkey != metadata.id.hash:
                    bt.logging.trace(
                        f"Sync for hotkey {hotkey} failed. Hash of content downloaded from hugging face {model.id.hash} "
                        + f"or the hash including the hotkey {hash_with_hotkey} do not match chain metadata {metadata}."
                    )
                    raise ValueError(
                        f"Sync for hotkey {hotkey} failed. Hash of content downloaded from hugging face does not match chain metadata. {metadata}"
                    )

            self.model_tracker.model_downloaded.add(hotkey)

    async def sync_model(self, hotkey: str) -> bool:
        """Updates local model for a hotkey if out of sync and returns if it was updated."""
        # Get the metadata for the miner.
        metadata = await self._get_metadata(hotkey)

        if not metadata:
            bt.logging.trace(
                f"No valid metadata found on the chain for hotkey {hotkey}"
            )
            return False

        if self.min_block and metadata.block < self.min_block:
            bt.logging.trace(
                f"Skipping model for {hotkey} since it was submitted at block {metadata.block} which is less than the minimum block {self.min_block}"
            )
            return False

        # Backwards compatability for models submitted before competition id added
        if metadata.id.competition_id is None:
            metadata.id.competition_id = constants.ORIGINAL_COMPETITION_ID

        parameters = ModelUpdater.get_competition_parameters(metadata.id.competition_id)
        if not parameters:
            bt.logging.trace(
                f"No competition parameters found for {metadata.id.competition_id}"
            )
            return False

        # Check what model id the model tracker currently has for this hotkey.
        tracker_model_metadata = self.model_tracker.get_model_metadata_for_miner_hotkey(
            hotkey
        )
        if metadata == tracker_model_metadata:
            return False

        # Get the local path based on the local store to download to (top level hotkey path)
        path = self.local_store.get_path(hotkey)

        # Otherwise we need to download the new model based on the metadata.
        model = await self.remote_store.download_model(metadata.id, path, parameters)

        # Check that the hash of the downloaded content matches.
        hash_matches_directly = model.id.hash == metadata.id.hash
        hash_with_hotkey = get_hash_of_two_strings(model.id.hash, hotkey)
        hash_matches_with_hotkey = hash_with_hotkey == metadata.id.hash

        if not (hash_matches_directly or hash_matches_with_hotkey):
            raise ValueError(
                f"Sync for hotkey {hotkey} failed. Hash of content downloaded from hugging face does not match chain metadata. {metadata}"
            )

        # Update the tracker
        self.model_tracker.on_miner_model_updated(hotkey, metadata)

        return True
