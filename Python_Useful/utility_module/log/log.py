#!/usr/bin/env python
# log
import logging
import logging.config

#logging.basicConfig(format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# disable logger from certain packages, like fbprophet or pandas
logger = logging.getLogger("fbprophet")
logger.setLevel(logging.WARNING)

# root logger of the main function
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.info("Start print log")
logger.debug("Do something")
logger.warning("Something maybe fail")
logger.info("Finish")
