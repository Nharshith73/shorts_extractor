# StreamClip AI — Current YouTube Source Diagnostic Report

**Report Date**: 2026-11-18  
**System**: StreamClip AI (Live Production Subsystem)  
**Target Watch URL**: `https://www.youtube.com/watch?v=lu613MYEzzY`  

---

## 1. Executive Summary

The production test source has been updated to YouTube livestream URL `https://www.youtube.com/watch?v=lu613MYEzzY`.  
All references in [source.yaml](file:///d:/clipsgta/shorts_extractor/source.yaml) and operational guides reference this active stream.

A live probe via official Google YouTube Data API v3 confirms that the broadcast is **CURRENTLY LIVE** and the live chat room is **ACTIVE** with messages actively streaming. However, raw video media ingestion into FFmpeg remains **BLOCKED** pending the provision of an authorized direct creator feed via `STREAMCLIP_LIVE_INPUT`.

---

## 2. Source Identification & Video Metadata

Retrieved via official endpoint:  
`https://www.googleapis.com/youtube/v3/videos?part=snippet,status,liveStreamingDetails&id=lu613MYEzzY`  
Authenticated securely using `X-Goog-Api-Key` HTTP header.

| Dimension | Value | Operational Context |
| :--- | :--- | :--- |
| **Video ID** | `lu613MYEzzY` | Active target video ID |
| **Video Title** | `GTA 6 TOMORROW🌴🌴RACKATHON 2 🌴 DAY 3🌴 BIG VILLA 🌴` | Live gaming broadcast |
| **Channel Title** | `Rachitroo` | Live broadcast creator |
| **Channel ID** | `UCsM3qWjsbJr5Kt7_Q1ulLjg` | Verified channel identifier |
| **Upload Status** | `uploaded` | Active live stream upload |
| **Privacy Status** | `public` | Publicly accessible broadcast |
| **Live Broadcast Content** | `live` | Currently active live stream |

---

## 3. Broadcast Lifecycle & Timing

| Lifecycle Field | Value | Notes |
| :--- | :--- | :--- |
| **Broadcast Status** | **`LIVE`** | Stream is actively transmitting |
| **is_live** | `True` | Live assertion verified |
| **Scheduled Start Time** | `2026-11-17T17:52:39Z` | Scheduled start |
| **Actual Start Time** | `2026-11-17T18:42:53Z` | Commenced live broadcast |
| **Actual End Time** | `None` | Ongoing live stream |

---

## 4. Live Chat Availability & Streaming Interface

| Chat Field | Value | Notes |
| :--- | :--- | :--- |
| **Chat Availability** | **`ACTIVE`** | Live chat room is open and operational |
| **Active Live Chat ID** | `Cg0KC2x1NjEzTVlFenpZKicKGFVDc00zcVdqc2JKcjVLdDdfUTF1bExqZxILbHU2MTNNWUV6elk` | Live chat room ID |
| **API Endpoint** | `https://www.googleapis.com/youtube/v3/liveChat/messages` | Official live chat endpoint |
| **Authentication** | `ACTIVE (Public-Data API via X-Goog-Api-Key)` | Fully authenticated |
| **Poll Rate Recommendation** | `~2.0 seconds` (`pollingIntervalMillis: 2029`) | Recommended poll window |
| **Live Message Capture** | **Verified Functional** | HTTP 200 with 10 messages returned per poll |

---

## 5. Media Acquisition Status & Architectural Boundaries

| Component | Status | Operational Details |
| :--- | :--- | :--- |
| **Acquisition Method** | `direct_media_input` | Configured in [source.yaml](file:///d:/clipsgta/shorts_extractor/source.yaml) |
| **Direct Media URL** | `UNCONFIGURED` | Neither `STREAMCLIP_LIVE_INPUT` nor `acquisition.media_url` is set |
| **Watch URL Role** | `WEB_METADATA_URL` | Used exclusively for YouTube API metadata and chat polling |
| **Direct Transport Stream** | **`NOT AVAILABLE`** | No FFmpeg raw stream input supplied |

> [!CRITICAL]
> **No Watch URL Ingestion to FFmpeg**:  
> A YouTube watch page (`https://www.youtube.com/watch?v=...`) is an HTML/JS web document. StreamClip AI does **NOT** pass watch URLs to FFmpeg and does **NOT** execute undocumented scrapers, token bypasses, or stream ripping utilities. Doing so violates compliance, access-control boundaries, and production reliability standards.

---

## 6. Exact Remaining Blocker

```
STREAMCLIP_LIVE_INPUT is unset and source.yaml acquisition.media_url is empty.
```

To transition from metadata/chat monitoring to live video capture:
The authorized creator must supply their direct media ingest stream (RTMP, SRT, or HLS), configured via:
```powershell
$env:STREAMCLIP_LIVE_INPUT = "rtmp://<ingest-server>/live/<stream-key>"
```
or directly in [source.yaml](file:///d:/clipsgta/shorts_extractor/source.yaml):
```yaml
acquisition:
  method: direct_media_input
  media_url: "rtmp://<ingest-server>/live/<stream-key>"
```

---

## 7. Status Triad: Readiness Classification

```
+-----------------------------------------------------------------------------+
|                     STREAMCLIP AI OPERATIONAL READINESS                     |
+--------------------------+-----------------------+--------------------------+
|      METADATA READY      |      CHAT READY       |  VIDEO INGESTION READY   |
+--------------------------+-----------------------+--------------------------+
|          [YES]           |         [YES]         |      [NO - BLOCKED]      |
|                          |                       |                          |
| Official YouTube Data    | Active chat poller    | Awaiting authorized      |
| API v3 connected.        | authenticated and     | creator direct media     |
| Broadcast confirmed      | ingesting real-time   | stream feed via          |
| LIVE from Rachitroo.     | live chat messages.   | STREAMCLIP_LIVE_INPUT.   |
+--------------------------+-----------------------+--------------------------+
```
