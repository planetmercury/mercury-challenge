# historian.py
# Historian classes for retrieving GSR events and warnings

__author__ = "Pete Haglich, peter.haglich@iarpa.gov"

from pandas import DataFrame
import numpy as np
from dateutil.parser import parse
import datetime

from mercury.common import db_management as db
from mercury.common.schema import JSONField, EventType, DiseaseType, CountryName

#
# Enumerate which of the methods in mercury.common.db_management should be
# used to access the Elasticsearch Scan API depending upon the document type
#
SCAN_METHOD = {db.GSR_TYPE:db.scanGSR, db.WARNING_TYPE:db.scanWarnings}

EVENT_DATE = JSONField.EVENT_DATE
EVENT_TYPE = JSONField.EVENT_TYPE
WARNING_ID = JSONField.WARNING_ID
PERF_ID = 'performer_id'
LOC_EVENT_TYPES = [EventType.CIVIL_UNREST, EventType.MILITARY_ACTION, EventType.NONSTATE_ACTOR]
cu_columns = [JSONField.COUNTRY, JSONField.STATE, JSONField.CITY, EVENT_DATE,
              JSONField.POPULATION, JSONField.REASON, JSONField.VIOLENT,
              JSONField.LATITUDE, JSONField.LONGITUDE]
ws_columns = [JSONField.COUNTRY, EVENT_DATE,
              JSONField.POPULATION, JSONField.REASON, JSONField.VIOLENT]
nsa_ma_columns = [JSONField.COUNTRY, JSONField.STATE, JSONField.CITY, EVENT_DATE,
                  JSONField.ACTOR, JSONField.TARGETS, JSONField.SUBTYPE,
                  JSONField.LATITUDE, JSONField.LONGITUDE]
rd_columns = [JSONField.COUNTRY, JSONField.STATE, JSONField.CITY, EVENT_DATE,
              JSONField.DISEASE,
              JSONField.LATITUDE, JSONField.LONGITUDE]
case_count_disease_columns = [JSONField.COUNTRY, EVENT_DATE, JSONField.DISEASE, JSONField.CASE_COUNT]
icews_columns = [JSONField.COUNTRY, EVENT_DATE, JSONField.CASE_COUNT]
germane_column_dict = {EventType.CIVIL_UNREST: cu_columns,
                       EventType.WIDESPREAD_CIVIL_UNREST: ws_columns,
                       EventType.DISEASE: rd_columns,
                       EventType.NONSTATE_ACTOR: nsa_ma_columns,
                       EventType.MILITARY_ACTION: nsa_ma_columns,
                       "Case_Count Disease": case_count_disease_columns,
                       EventType.ICEWS_PROTEST: icews_columns}
index_column_dict = {
  db.GSR_TYPE: [JSONField.EVENT_ID, JSONField.EARLIEST_REPORTED_DATE],
  db.WARNING_TYPE: [WARNING_ID, JSONField.PROBABILITY, JSONField.TIMESTAMP,
                    JSONField.SEQUENCE, JSONField.LATEST],
  }


class Historian:
    """Retrieves events or warnings for use by Scorer or baserate model"""
    REQUIRE_LOCALITY = False

    def __init__(self, locality=False):
        """
        :param locality: Optional boolean indicating whether city and state
          attributes are required for retrieving events (default False)
        """
        self.REQUIRE_LOCALITY = locality

    def _query_base(self, country):
        """
        Builds a base query
        :param country: Which country
        :param event_type: Which event type
        :return: dict with base query
        """
        q1 = {'query': {}}
        q1["query"]["bool"] = {}
        must = [{"term": {JSONField.COUNTRY: country}}]
        if self.REQUIRE_LOCALITY:
          must += [
            {"exists" : {"field" : JSONField.STATE}},
            {"exists" : {"field" : JSONField.CITY}},
            ]
        q1["query"]["bool"]["must"] = must

        return q1

    def _format_response(self, doc_type, history, event_type):
        """
Given the specified document type, collection of documents retrieved from the
database, and the specified event type, filter the database retrieval down to
just the germane columns.
        """

        if event_type and history is not None:
            germane_columns = [EVENT_TYPE] + germane_column_dict[event_type] \
                              + index_column_dict[doc_type]
            if event_type in LOC_EVENT_TYPES:
                if doc_type == db.GSR_TYPE:
                    germane_columns += [JSONField.APPROXIMATE_LOCATION]
            history = history[germane_columns]

        return history

    def get_history(self, doc_type, country, event_type=False,
                    start_date=False, end_date = False,
                    verbose=False, performer_id=None, **matchargs):
        """
        Retrieves the history from the index
        :param doc_type: Which document type from the index
        :param event_type: Which type of event
        :param country: The country for the data
        :param start_date: First date
        :param end_date: End date
        :param verbose: Whether to print details
        :param matchargs: Additional match terms for the query

        :returns: DataFrame of event data
        """

        q1 = self._query_base(country)

        if event_type:
            q1["query"]["bool"]["must"].append({"term": {EVENT_TYPE: event_type}})

        if performer_id and doc_type == db.WARNING_TYPE:
            q1["query"]["bool"]["must"].append({"term": {PERF_ID: performer_id}})

        for k in matchargs:
            q1["query"]["bool"]["must"].append({"match": {k: matchargs[k]}})

        if start_date or end_date:
            range_part = {"range":
                              {EVENT_DATE:
                                   {}
                               }
                          }
            if start_date:
                range_part["range"][EVENT_DATE]["gte"] = start_date
            if end_date:
                range_part["range"][EVENT_DATE]["lte"] = end_date

            q1["query"]["bool"]["must"].append(range_part)

        if verbose:
            print(q1)

        resp_list = [i['_source'] for i in SCAN_METHOD[doc_type](query=q1)]
        if not resp_list:
          if verbose: print("No events matching these parameters")
          history = None
        else:
          history = DataFrame(resp_list).sort_values(EVENT_DATE)
          if verbose:
            print("There were {0} hits".format(len(resp_list)))
            print(resp_list)
        if history is not None:
            history = self._format_response(doc_type, history, event_type)
        return history

    def get_concise_warnings(country, event_type=False,
                             start_date=False, end_date = False,
                             verbose=False, performer_id=None, **matchargs):
        pass

    def get_age(self, item, old_hist_df):
        """
        Adds the length of time since there was an event in that location
        :param old_hist_end: Last date for the old history
        :param item: Series with event or warning data
        :param old_hist_start: First date we want to consider for the old history
        :return: DataFrame
        """

        if old_hist_df is None:
            age = np.nan
        else:
            cc = item["Country"]
            new_date = parse(item["Event_Date"])
            old_collocated = old_hist_df[old_hist_df.Latitude == item["Latitude"]]
            old_collocated = old_collocated[old_collocated.Longitude == item["Longitude"]]
            if len(old_collocated) > 0:
                old_collocated.Event_Date = old_collocated.Event_Date.apply(parse)
                old_item = old_collocated.ix[old_collocated.Event_Date.argmax()]
                age = new_date - old_item["Event_Date"]
                age = age.days
            else:
                age = np.nan

        return age

    def add_ages(self, new_hist, old_hist_df):
        """
        Adds age column to new_hist DataFrame
        :param new_hist: DataFrame with new history
        :param old_hist_start: First date of old history
        :return: DataFrame with age added
        """
        out_hist = new_hist.copy()
        new_hist_start = new_hist.Event_Date.apply(parse).min()
        out_hist["Age"] = out_hist.apply(self.get_age, old_hist_df=old_hist_df,
                                         axis=1)
        return out_hist


    @staticmethod
    def to_kml(the_df, facet_column=None):
        """
        Converts a data frame of warnings or events to KML format.
        :param the_df: DataFrame with Latitude, Longitude, and ID column
        :return:
        """
        if "Event_ID" in the_df.columns:
            the_df["ID"] = the_df.Event_ID
        elif "Warning_ID" in the_df.columns:
            the_df["ID"] = the_df.Warning_ID
        else:
            the_df["ID"] = the_df.index
        kml = ['<kml xmlns="http://www.opengis.net/kml/2.2"><Document>']

        def row_to_xml(row, indent_level=2):
            the_id = row.ix["ID"]
            evt_type = row.ix["Event_Type"]
            xml = ['<Placemark id="{0}"><name>{0}</name>'.format(the_id)]
            if evt_type == "Civil Unrest":
                if row.ix["Violent"] == "True":
                    xml.append("<Style><IconStyle><Icon>")
                    xml.append("<href>http://maps.google.com/mapfiles/kml/paddle/red-stars.png</href>")
                    xml.append("</Icon></IconStyle></Style>")
                xml.append("<ExtendedData>")
                xml.append("<Data name='Population'><value>{0}</value></Data>".format(row.ix["Population"]))
                xml.append("<Data name='Reason'><value>{0}</value></Data>".format(row.ix["Reason"]))
            elif evt_type in ["Military Action", "Non-State Actor"]:
                es = row.ix["Event_Subtype"]
                if es in ["Armed Conflict", "Armed Assault"]:
                    xml.append("<Style><IconStyle><Icon>")
                    xml.append("<href>http://maps.google.com/mapfiles/kml/paddle/red-stars.png</href>")
                    xml.append("</Icon></IconStyle></Style>")
                elif es == "Bombing":
                    xml.append("<Style><IconStyle><Icon>")
                    xml.append("<href>http://maps.google.com/mapfiles/kml/shapes/volcano.png</href>")
                    xml.append("</Icon></IconStyle></Style>")
                elif es == "Hostage Taking":
                    xml.append("<Style><IconStyle><Icon>")
                    xml.append("<href>http://maps.google.com/mapfiles/kml/shapes/fishing.png</href>")
                    xml.append("</Icon></IconStyle></Style>")
                elif es == "Force Posture":
                    xml.append("<Style><IconStyle><Icon>")
                    xml.append("<href>http://maps.google.com/mapfiles/kml/shapes/police.png</href>")
                    xml.append("</Icon></IconStyle></Style>")
                xml.append("<ExtendedData>")
                xml.append("<Data name='Actor'><value>{0}</value></Data>".format(row.ix["Actor"]))
                try:
                    targets = row.ix["Targets"]
                    target_list = [(t["Target_Status"], t["Target"]) for t in targets]
                    for t in target_list:
                        xml.append("<Data name='Target'><value>Status: {0}, Target:{1}</value></Data>".format(*t))
                except TypeError:
                    print(row.ix["Targets"])
                xml.append("<Data name='Event Subtype'><value>{0}</value></Data>".format(row.ix["Event_Subtype"]))
            if "Age" in row.index:
                if np.isnan(row.ix["Age"]):
                    age_val = "Virgin"
                    xml.append("<Data name='Age'><value>{0}</value></Data>".format(age_val))
                else:
                    age_val = row.ix["Age"]
                    xml.append("<Data name='Age'><value>{0:.0f} Days</value></Data>".format(age_val))
            xml.append("</ExtendedData>")
            xml.append('<Point id="{0}"><name>{0}</name>'.format(the_id))
            lat, long = (row.ix["Latitude"], row.ix["Longitude"])
            xml.append("<coordinates>{0},{1}</coordinates>".format(long, lat))
            xml.append("</Point>")
            xml.append("</Placemark>")
            return "\n".join(xml)

        the_countries = sorted(the_df.Country.unique())
        for cc in the_countries:
            kml.append("<Folder><name>{0}</name>".format(cc))
            cc_df = the_df[the_df.Country == cc]
            if facet_column is not None:
                facet_values = cc_df[facet_column].unique()
                for fv in sorted(facet_values):
                    kml.append('<Folder><name>{0}</name>'.format(fv))
                    f_df = cc_df[cc_df[facet_column] == fv]
                    fv_result = f_df.apply(row_to_xml, axis=1)
                    kml.append("".join(fv_result))
                    kml.append("</Folder>")
            else:
                result = "".join(the_df.apply(row_to_xml, axis=1))
                kml.append(result)
            kml.append("</Folder>")
        kml.append('</Document></kml>')
        kml = "".join(kml)

        return kml


class MaNsaHistorian(Historian):

    def __init__(self):
        super().__init__(locality=True)

    def _format_response(self, doc_type, history, event_type):
        """
        Given the specified document type, collection of documents retrieved from the
        database, and the specified event type, filter the database retrieval down to
        just the germane columns.
        """

        if event_type and history is not None:
            germane_columns = [EVENT_TYPE] + germane_column_dict[event_type] \
                              + index_column_dict[doc_type]
            if event_type in LOC_EVENT_TYPES:
                if doc_type == db.GSR_TYPE:
                    germane_columns += [JSONField.APPROXIMATE_LOCATION]
            history = history[germane_columns]
            history[JSONField.GSR_TARGET_STATUS] = history[JSONField.TARGETS].apply(lambda x:
                                                                                    [item[JSONField.GSR_TARGET_STATUS]
                                                                                     for item in x])
            history[JSONField.GSR_TARGET] = history[JSONField.TARGETS].apply(lambda x:
                                                                    [item[JSONField.GSR_TARGET] for item in x])
            history.drop(JSONField.TARGETS, axis=1, inplace=True)

        return history


class RareDiseaseHistorian(Historian):
    """Specialty class to retrieve rare disease events.  The query is a bit complex, so
    just passing kwargs to Historian won't work"""

    def __init__(self):
        super().__init__(locality=True)

    def _query_base(self, country):
        """
        Retrieves rare disease history for the given country
        :param index: Which index to query
        :param country: Country to use
        :param start_date: First date to use, default false
        :param end_date: Last date, default False
        :param verbose: Print out gory details?
        :param matchargs: Other arguments to be passed to the query
        :return:
        """
        q1 = {'query': {}}
        q1["query"]["bool"] = {"must": [], "must_not": []}
        part_0a = {"term": {JSONField.COUNTRY: country}}
        part_0b = {"term": {EVENT_TYPE: EventType.DISEASE}}
#        q1["query"]["bool"]["must"].append(part_0a)
#        q1["query"]["bool"]["must"].append(part_0b)
        q1["query"]["bool"]["must"] += [
          part_0a,
          part_0b,
          {"exists" : {"field" : JSONField.STATE}},
          {"exists" : {"field" : JSONField.CITY}},
          ]

        part_1a = {"term": {JSONField.DISEASE: DiseaseType.MERS}}
        part_1b = {"term": {JSONField.COUNTRY: CountryName.SAUDI_ARABIA}}
        part_1 = {"bool": {"must": [part_1a, part_1b]}}
        q1["query"]["bool"]["must_not"].append(part_1)
        part_2a = {"term": {JSONField.DISEASE: DiseaseType.AVIAN_INFLUENZA}}
        part_2b = {"term": {JSONField.COUNTRY: CountryName.EGYPT}}
        part_2 = {"bool": {"must": [part_2a, part_2b]}}
        q1["query"]["bool"]["must_not"].append(part_2)

        return q1

    def get_history(self, doc_type, country, start_date=False,
                    end_date=False, verbose=False, performer_id=None, *args, **matchargs):
        """
        Specialized override of get_history to predefine the event type
        :param doc_type: Which document type from the index
        :param country: Country to search
        :param start_date: First date
        :param end_date: Last date
        :param verbose: Gory details?
        :param matchargs: Additional arguments
        :return: history DataFrame
        """
        if 'event_type' in matchargs:
            del matchargs['event_type']
        history_df = super().get_history(doc_type, country=country, event_type=EventType.DISEASE,
                                         start_date=start_date, end_date=end_date,
                                         verbose=verbose, performer_id=performer_id, **matchargs)

        history_df = self._format_response(doc_type=doc_type, history=history_df, event_type=EventType.DISEASE)
        return history_df

    def _format_response(self, doc_type, history, event_type=EventType.DISEASE):
        """
        Given the specified document type, collection of documents retrieved from the
        database, and the specified event type, filter the database retrieval down to
        just the germane columns.
        """

        if history is not None:
            germane_columns = [EVENT_TYPE] + germane_column_dict[event_type] \
                              + index_column_dict[doc_type]
            if doc_type == db.GSR_TYPE:
                germane_columns += [JSONField.APPROXIMATE_LOCATION]
            history = history[germane_columns]

        return history


class CaseCountDiseaseHistorian(Historian):
    """
    Special purpose Historian subclass for the two case count diseases
    """

    def _query_base(self, country):
        """
        Retrieves rare disease history for the given country
        :param country: Country to use
        :return:
        """
        q1 = {'query': {}}
        q1["query"]["bool"] = {"must": []}
        part_0a = {"term": {JSONField.COUNTRY: country}}
        part_0b = {"term": {EVENT_TYPE: EventType.DISEASE}}
        q1["query"]["bool"]["must"].append(part_0a)
        q1["query"]["bool"]["must"].append(part_0b)

        part_1and2 = {"bool": {"should": []}}
        part_1a = {"term": {JSONField.DISEASE: DiseaseType.MERS}}
        part_1b = {"term": {JSONField.COUNTRY: CountryName.SAUDI_ARABIA}}
        part_1 = {"bool": {"must": [part_1a, part_1b]}}
        part_1and2["bool"]["should"].append(part_1)
        part_2a = {"term": {JSONField.DISEASE: DiseaseType.AVIAN_INFLUENZA}}
        part_2b = {"term": {JSONField.COUNTRY: CountryName.EGYPT}}
        part_2 = {"bool": {"must": [part_2a, part_2b]}}
        part_1and2["bool"]["should"].append(part_2)
        q1["query"]["bool"]["must"].append(part_1and2)

        return q1

    def get_history(self, doc_type, country, start_date=False,
                    end_date=False, verbose=False, performer_id=None, **matchargs):
        """
        Specialized override of get_history to predefine the event type
        :param doc_type: Which document type from the index
        :param country: Country to search
        :param start_date: First date
        :param end_date: Last date
        :param verbose: Gory details?
        :param matchargs: Additional arguments
        :return: history DataFrame
        """
        history_df = super().get_history(doc_type, country=country,
                                         event_type=EventType.DISEASE,
                                         start_date=start_date, end_date=end_date,
                                         verbose=verbose, performer_id=performer_id, **matchargs)
        return history_df

    def _format_response(self, doc_type, history, event_type=EventType.DISEASE):
        """
Given the specified document type, collection of documents retrieved from the
database, and the specified event type, filter the database retrieval down to
just the germane columns.
        """

        if history is not None:
            germane_columns = [EVENT_TYPE] + germane_column_dict["Case_Count Disease"] \
                              + index_column_dict[doc_type]
            try:
                germane_columns.remove(JSONField.PROBABILITY)
            except ValueError:
                pass
            history = history[germane_columns]

        return history


class IcewsHistorian(Historian):
    """
    Special purpose Historian subclass for the two case count diseases
    """

    def get_history(self, doc_type, country, start_date=False,
                    end_date=False, verbose=False, performer_id=None, **matchargs):
        """
        Specialized override of get_history to predefine the event type
        :param doc_type: Which document type from the index
        :param country: Country to search
        :param start_date: First date
        :param end_date: Last date
        :param verbose: Gory details?
        :param matchargs: Additional arguments
        :return: history DataFrame
        """
        if 'event_type' in matchargs:
            del matchargs['event_type']

        history_df = super().get_history(doc_type, country=country,
                                         event_type=EventType.ICEWS_PROTEST,
                                         start_date=start_date, end_date=end_date,
                                         verbose=verbose, performer_id=performer_id, **matchargs)
        return history_df

    def _format_response(self, doc_type, history, event_type=EventType.ICEWS_PROTEST):
        """
Given the specified document type, collection of documents retrieved from the
database, and the specified event type, filter the database retrieval down to
just the germane columns.
        """

        index_col_dict = {
          db.GSR_TYPE: [JSONField.EVENT_ID],
          db.WARNING_TYPE: [WARNING_ID, JSONField.TIMESTAMP],
          }

        if history is not None:
            germane_columns = [EVENT_TYPE] + germane_column_dict[event_type] \
                              + index_col_dict[doc_type]
            try:
                germane_columns.remove(JSONField.PROBABILITY)
            except ValueError:
                pass
            #print(germane_columns)
            history = history[germane_columns]
            #print(history)

        return history

def germane_countries(
  event_type, start_date, end_date,
  index=db.INDEX_NAME, doc_type=db.GSR_TYPE,
  **kwargs):
    """
    Determines the countries that had an event or warning of that type in
    the time period given
    :param event_type: Which event type
    :param start_date: First date
    :param end_date: Last date
    :param index: Which index to search, default is "mercury"
    :param doc_type: Which document type to search, default is "gsr"
    :param kwargs: Optional search arguments
    :return:
    """
    q1 = {"query":
            {"bool": {
               "must": [
                {"match": {EVENT_TYPE: event_type}}
            ]
                }
            },
          }
    range_part = {"range":
                     {EVENT_DATE:
                        {}
                     }
                 }
    range_part["range"][EVENT_DATE]["gte"] = start_date
    range_part["range"][EVENT_DATE]["lte"] = end_date
    q1["query"]["bool"]["must"].append(range_part)
    for kwa in kwargs:
        q1["query"]["bool"]["must"].append({"match": {kwa: kwargs[kwa]}})
    result = [i['_source'] for i in SCAN_METHOD[doc_type](query=q1)]
    if not result: return []
    return sorted(set([i[JSONField.COUNTRY] for i in result]))

